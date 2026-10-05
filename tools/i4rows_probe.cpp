// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
//
// i4rows_probe.cpp -- INT4 GEMV over M = 1..4 activation rows (speculative
// verify) on Qwen3.8-27B's dense shapes: time per call and weight GB/s of
// the engine's kernels (decode GEMV at M = 1, the w4a16 DPAS small-M GEMM
// from gemv_decode.cpp and from libgrimoire_gemm.so) and of local variants:
// the DPAS kernel under other split-K plans (128 GRF), and an fp32 FMA
// kernel over the rows (ROWS=1; ALU-bound, ~250 GB/s at M = 4).  The
// weights rotate over enough copies to stream from DRAM (no L2 reuse).
// Argument: index of one shape (default: all).
//
// Build (grimoire-dev, from the repo root):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -fno-fast-math \
//     -ffp-contract=fast -fno-math-errno -fsycl-device-code-split=per_kernel \
//     -I include -I src tools/i4rows_probe.cpp src/gemv_decode.cpp \
//     -Lbin -lgrimoire_gemm '-Wl,-rpath,$ORIGIN' -o bin/i4rows_probe
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/prefetch.hpp>
#include <algorithm>
#include "b70/weights.hpp"
#include "kernels.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

using namespace b70;
namespace es = sycl::ext::intel::esimd;

namespace {
constexpr int kMaxKS = 64;

// V: 0 = full kernel, 1 = loads only (no decode / FMA: the access pattern's
// streaming rate), 2 = decode only (no FMA over rows)
template <int R, int GS, int MB, int V>
sycl::event rows_k(sycl::queue& q, const QuantWeight& w, const float* x, float* y,
                   int mb, int KP) {
    constexpr int RP = R * MB;
    constexpr int NG = 128 / GS;
    constexpr int BPG = GS / 2;
    static_assert(R * NG <= 16, "one 16-lane gather per step");
    const int K = w.K, N = w.N, KS = K / KP;
    const uint8_t* pay = w.payload;
    const uint16_t* scl = static_cast<const uint16_t*>(w.scales);
    const uint8_t* zer = w.zeros;
    const int64_t rb = w.row_bytes;
    const uint32_t rs = uint32_t(w.row_scales);
    const int n_wg = N / R;
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(size_t(n_wg) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            es::slm_init<kMaxKS * RP * 4>();
            const int t = int(it.get_local_id(0));
            const int64_t row0 = int64_t(it.get_group(0)) * R;
            const int kb = t * KP;
            es::simd<uint32_t, 16> gi;
            #pragma unroll
            for (int i = 0; i < 16; ++i)
                gi[i] = uint32_t(row0 + i % R) * rs + uint32_t((i / R) % NG) + uint32_t(kb / GS);
            es::simd<uint8_t, 64> pb[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) pb[r] = es::block_load<uint8_t, 64>(pay + (row0 + r) * rb + kb / 2);
            es::simd<uint16_t, 16> sv = es::gather<uint16_t, 16>(scl, gi * 2u);
            es::simd<uint8_t, 16> zv = es::gather<uint8_t, 16>(zer, gi);
            es::simd<float, 16> acc[R][MB];
            #pragma unroll
            for (int r = 0; r < R; ++r)
                #pragma unroll
                for (int m = 0; m < MB; ++m) acc[r][m] = 0.0f;
            for (int k = kb; k < kb + KP; k += 128) {
                es::simd<float, 64> xe[MB], xo[MB];
                #pragma unroll
                for (int m = 0; m < MB; ++m) {
                    if (m < mb) {
                        es::simd<float, 128> xs = es::block_load<float, 128>(x + int64_t(m) * K + k);
                        xe[m] = xs.template select<64, 2>(0);
                        xo[m] = xs.template select<64, 2>(1);
                    } else {
                        xe[m] = 0.0f;
                        xo[m] = 0.0f;
                    }
                }
                const bool more = k + 128 < kb + KP;
                es::simd<uint8_t, 64> pn[R];
                es::simd<uint16_t, 16> sn;
                es::simd<uint8_t, 16> zn;
                if (more) {
                    #pragma unroll
                    for (int r = 0; r < R; ++r)
                        pn[r] = es::block_load<uint8_t, 64>(pay + (row0 + r) * rb + (k + 128) / 2);
                    gi += uint32_t(NG);
                    sn = es::gather<uint16_t, 16>(scl, gi * 2u);
                    zn = es::gather<uint8_t, 16>(zer, gi);
                }
                if constexpr (V == 1) {
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        es::simd<uint16_t, 64> u = pb[r];
                        es::simd<float, 16> f = u.template select<16, 1>(0) + u.template select<16, 1>(48);
                        f += float(sv[r]) + float(zv[r]);
                        #pragma unroll
                        for (int m = 0; m < MB; ++m)
                            acc[r][m] += f * (xe[m].template select<16, 1>(0) + xo[m].template select<16, 1>(48));
                    }
                } else {
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        #pragma unroll
                        for (int g = 0; g < NG; ++g) {
                            const uint32_t zu = uint32_t(zv[g * R + r]);
                            const uint32_t sg = (zu + 1u) >> 8;
                            const uint16_t xm = uint16_t(sg * 0x88u);
                            const sycl::half zh = sycl::bit_cast<sycl::half>(uint16_t(0x6400u | (zu - sg * 247u)));
                            const float s = sycl::bit_cast<float>(uint32_t(sv[g * R + r]) << 16);
                            es::simd<uint16_t, BPG> u = pb[r].template select<BPG, 1>(g * BPG);
                            u ^= xm;
                            es::simd<uint16_t, BPG> lb = (u & 0xF) | 0x6400;
                            es::simd<uint16_t, BPG> hb = (u >> 4) | 0x6400;
                            es::simd<sycl::half, BPG> lh = lb.template bit_cast_view<sycl::half>().read() - zh;
                            es::simd<sycl::half, BPG> hh = hb.template bit_cast_view<sycl::half>().read() - zh;
                            es::simd<float, BPG> wl = lh, wh = hh;
                            if constexpr (V == 2) {
                                es::simd<float, 16> part = 0.0f;
                                #pragma unroll
                                for (int b = 0; b < BPG / 16; ++b)
                                    part += wl.template select<16, 1>(16 * b) + wh.template select<16, 1>(16 * b);
                                #pragma unroll
                                for (int m = 0; m < MB; ++m) acc[r][m] += part * s * xe[m].template select<16, 1>(0);
                            } else {
                                #pragma unroll
                                for (int m = 0; m < MB; ++m) {
                                    es::simd<float, 16> part = 0.0f;
                                    #pragma unroll
                                    for (int b = 0; b < BPG / 16; ++b)
                                        part += wl.template select<16, 1>(16 * b) * xe[m].template select<16, 1>(g * BPG + 16 * b) +
                                                wh.template select<16, 1>(16 * b) * xo[m].template select<16, 1>(g * BPG + 16 * b);
                                    acc[r][m] += part * s;
                                }
                            }
                        }
                    }
                }
                if (more) {
                    #pragma unroll
                    for (int r = 0; r < R; ++r) pb[r] = pn[r];
                    sv = sn;
                    zv = zn;
                }
            }
            es::simd<float, RP> red;
            #pragma unroll
            for (int r = 0; r < R; ++r)
                #pragma unroll
                for (int m = 0; m < MB; ++m) red[r * MB + m] = es::reduce<float>(acc[r][m], std::plus<>());
            if (KS > 1) {
                es::slm_block_store<float, RP>(t * RP * 4, red);
                es::barrier();
                if (t != 0) return;
                red = 0.0f;
                for (int j = 0; j < KS; ++j) red += es::slm_block_load<float, RP>(j * RP * 4);
            }
            #pragma unroll
            for (int m = 0; m < MB; ++m) {
                if (m >= mb) continue;
                #pragma unroll
                for (int r = 0; r < R; ++r) y[int64_t(m) * N + row0 + r] = red[r * MB + m];
            }
        });
    });
}

struct SmallmPlan { int KS, kc, TPT; };
SmallmPlan plan_x(int N, int K, int ksmul) {
    const int tiles = N / 16;
    int KS = 1;
    if (ksmul < 0) {
        KS = -ksmul;                              // explicit
    } else if (ksmul == 0) {                      // rule: ~12288 threads, kc >= 256, KS >= 2
        KS = std::max(2, (12288 + tiles - 1) / tiles);
        KS = std::min(KS, std::max(1, K / 256));
    } else {
    while (KS < 16 && tiles * KS < 4096 && K / (KS * 2) >= 128) KS *= 2;
    KS = std::min(64, KS * ksmul);
    }
    int kc = (K + KS - 1) / KS;
    kc = (kc + 127) / 128 * 128;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= 64) TPT *= 2;
    return {KS, kc, TPT};
}
template <int RBN, int GS, int DEC = 1>
sycl::event sm_local(sycl::queue& q, const QuantWeight& w, const sycl_bf16* X,
                             float* Y, int M, int ksmul) {
    const std::vector<sycl::event> deps;
    static_assert(GS == 64 || GS == 128, "INT4 group of 64 or 128");
    constexpr int CT = 16;
    const int N = w.N, K = w.K;
    const SmallmPlan p = plan_x(N, K, ksmul);
    const int KS = p.KS, kc = p.kc, TPT = p.TPT;
    const int tiles = N / CT;
    const int groups = (tiles + TPT - 1) / TPT;
    const uint32_t* payw = reinterpret_cast<const uint32_t*>(w.payload);
    const uint16_t* sclh = static_cast<const uint16_t*>(w.scales);
    const uint8_t* zer = w.zeros;
    const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(N) - 1, PP = unsigned(w.row_bytes) - 1;
    const unsigned RS = unsigned(w.row_scales);
    const unsigned XW = unsigned(K) * 2 - 1, XH = unsigned(M) - 1;
    const unsigned YW = unsigned(N) * 4 - 1;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(groups) * TPT * KS, size_t(TPT) * KS),
                       [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            namespace es = sycl::ext::intel::esimd;
            namespace xmx = sycl::ext::intel::esimd::xmx;
            es::slm_init<64 * RBN * 128 * 4>();
            const int lid = int(it.get_local_id(0));
            const int tw = lid / KS, ks = lid % KS;
            const int tile = int(it.get_group(0)) * TPT + tw;
            const bool live = tile < tiles;
            const int n0 = (live ? tile : 0) * CT;
            const int kb = ks * kc;
            const int ke = kb + kc < K ? kb + kc : K;
            constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
                es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
            // element index of each column's group-0 scale / zero
            const es::simd<uint32_t, 16> gbase =
                (es::simd<uint32_t, 16>(0, 1) + uint32_t(n0)) * RS;
            es::simd<float, RBN * 128> acc = 0.0f;
            if (live) {
                // Each column's scale / zero for the next 128 K are gathered one
                // step ahead, like the payload prefetch: gathered in-step they
                // were a dependent load in front of every decode (Qwen3.8-27B
                // gate_up at M = 4 ran at ~380 GB/s against ~550 for MXFP4).
                constexpr int NGS = 128 / GS;        // groups per 128 K
                es::simd<uint32_t, 16> zn[NGS], sn[NGS];
                #pragma unroll
                for (int g = 0; g < NGS; ++g) {
                    const es::simd<uint32_t, 16> gi = gbase + uint32_t((kb + GS * g) / GS);
                    zn[g] = es::gather<uint8_t, 16>(zer, gi);
                    sn[g] = es::gather<uint16_t, 16>(sclh, gi * 2u);
                }
                for (int k = kb; k < ke; k += 128) {
                    if (k + 128 < ke)
                        es::prefetch_2d<uint32_t, 16, CT>(payw, PW, PH, PP, (k + 128) / 8, n0, PFH);
                    es::simd<uint32_t, 8 * CT> twh[2];
                    twh[0] = es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8, n0);
                    twh[1] = es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8 + 8, n0);
                    es::simd<uint32_t, 16> zc[NGS], scb[NGS];
                    #pragma unroll
                    for (int g = 0; g < NGS; ++g) { zc[g] = zn[g]; scb[g] = sn[g]; }
                    if (k + 128 < ke) {
                        #pragma unroll
                        for (int g = 0; g < NGS; ++g) {
                            const es::simd<uint32_t, 16> gi = gbase + uint32_t((k + 128 + GS * g) / GS);
                            zn[g] = es::gather<uint8_t, 16>(zer, gi);
                            sn[g] = es::gather<uint16_t, 16>(sclh, gi * 2u);
                        }
                    }
                    es::simd<uint32_t, 16> xm;
                    es::simd<float, 16> zf, sc;
                    #pragma unroll
                    for (int hf = 0; hf < 2; ++hf) {
                        if (hf == 0 || GS == 64) {
                            es::simd<uint32_t, 16> zu = zc[GS == 64 ? hf : 0];
                            es::simd<uint32_t, 16> sb = scb[GS == 64 ? hf : 0];
                            // zero 0xff -> signed s4: flip every nibble's bit 3, zero 8
                            es::simd<uint32_t, 16> sg = (zu + 1u) >> 8;
                            xm = sg * 0x88888888u;
                            es::simd<uint32_t, 16> zb = (zu - sg * 247u) | 0x4B000000u;
                            zf = zb.template bit_cast_view<float>().read();
                            es::simd<uint32_t, 16> sw = sb << 16;
                            sc = sw.template bit_cast_view<float>().read();
                        }
                        es::simd<float, 128> tmp[RBN];
                        #pragma unroll
                        for (int rb = 0; rb < RBN; ++rb) tmp[rb] = 0.0f;
                        #pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            const int kk = k + 64 * hf + 32 * b;
                            es::simd<sycl_bf16, 128> a0[RBN], a1[RBN];
                            #pragma unroll
                            for (int rb = 0; rb < RBN; ++rb) {
                                a0[rb] = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk, rb * 8);
                                a1[rb] = es::load_2d<sycl_bf16, 16, 8>(X, XW, XH, XW, kk + 16, rb * 8);
                            }
                            es::simd<uint32_t, 128> vb[2];
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                es::simd<uint32_t, 16> wv = twh[hf].template select<16, 1>((4 * b + j) * CT);
                                wv ^= xm;
                                #pragma unroll
                                for (int qb = 0; qb < 4; ++qb) {
                                    es::simd<uint32_t, 16> tb = qb ? (wv >> (8 * qb)) : wv;
                                    es::simd<uint32_t, 16> pk;
                                    if constexpr (DEC == 0) {
                                        pk = tb;
                                    } else if constexpr (DEC == 2) {
                                        // bf16 magic 128 + q, zero folded out (timing only)
                                        pk = (tb & 0xFu) | ((tb << 12) & 0xF0000u) | 0x43004300u;
                                    } else {
                                    es::simd<uint32_t, 16> lo = (tb & 0xFu) | 0x4B000000u;
                                    es::simd<uint32_t, 16> hi = ((tb >> 4) & 0xFu) | 0x4B000000u;
                                    es::simd<float, 16> lf = lo.template bit_cast_view<float>().read() - zf;
                                    es::simd<float, 16> hv = hi.template bit_cast_view<float>().read() - zf;
                                    pk =
                                        (lf.template bit_cast_view<uint32_t>().read() >> 16) |
                                        (hv.template bit_cast_view<uint32_t>().read() & 0xFFFF0000u);
                                    }
                                    const int kp = 4 * j + qb;
                                    vb[kp >> 3].template select<16, 1>((kp & 7) * 16) = pk;
                                }
                            }
                            #pragma unroll
                            for (int rb = 0; rb < RBN; ++rb) {
                                tmp[rb] = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                    tmp[rb], vb[0].template bit_cast_view<sycl_bf16>().read(), a0[rb]);
                                tmp[rb] = xmx::dpas<8, 8, float, float, sycl_bf16, sycl_bf16>(
                                    tmp[rb], vb[1].template bit_cast_view<sycl_bf16>().read(), a1[rb]);
                            }
                        }
                        #pragma unroll
                        for (int rb = 0; rb < RBN; ++rb)
                            #pragma unroll
                            for (int r = 0; r < 8; ++r)
                                acc.template select<16, 1>(rb * 128 + 16 * r) +=
                                    tmp[rb].template select<16, 1>(16 * r) * sc;
                    }
                }
            }
            if (KS > 1) {
                es::slm_block_store<float, RBN * 128>(lid * RBN * 128 * 4, acc);
                es::barrier();
                if (ks != 0 || !live) return;
                for (int i = 1; i < KS; ++i)
                    acc += es::slm_block_load<float, RBN * 128>((tw * KS + i) * RBN * 128 * 4);
            } else if (!live) {
                return;
            }
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb)
                es::store_2d<float, 16, 8>(Y, YW, XH, YW, n0, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>(rb * 128)));
        });
    });
}


int kp_for(int K) {
    for (int kp = 128; kp <= K; kp += 128)
        if (K % kp == 0 && K / kp <= kMaxKS) return kp;
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    struct Shape { const char* nm; int N, K; };
    std::vector<Shape> shapes = {{"gate_up", 34816, 5120}, {"down", 5120, 17408}, {"dn_qkv", 10240, 5120},
                                 {"z", 6144, 5120}, {"out", 5120, 6144}, {"lm_head", 248320, 5120},
                                 {"attn_q", 12288, 5120}, {"kv", 1024, 5120}};
    const int only = argc > 1 ? std::atoi(argv[1]) : -1;
    std::mt19937 rng(7);
    for (int si = 0; si < int(shapes.size()); ++si) {
        if (only >= 0 && only != si) continue;
        const Shape sh = shapes[si];
        const int N = sh.N, K = sh.K, GS = 128, rsc = K / GS;
        const size_t pb = size_t(N) * K / 2, sb = size_t(N) * rsc * 2, zb = size_t(N) * rsc;
        const size_t wb = pb + sb + zb;
        const int copies = std::max(1, std::min(8, int((1536ull << 20) / wb)));
        std::vector<uint8_t> hp(pb), hz(zb);
        std::vector<uint16_t> hs(size_t(N) * rsc);
        for (auto& v : hp) v = uint8_t(rng());
        for (auto& v : hz) v = uint8_t(rng() % 16);
        for (auto& v : hs) v = sycl::bit_cast<uint16_t>(sycl_bf16(0.001f * float(1 + rng() % 16)));
        std::vector<QuantWeight> ws(copies);
        std::vector<void*> allocs;
        for (int c = 0; c < copies; ++c) {
            uint8_t* p = sycl::malloc_device<uint8_t>(pb, q);
            uint16_t* s = sycl::malloc_device<uint16_t>(size_t(N) * rsc, q);
            uint8_t* z = sycl::malloc_device<uint8_t>(zb, q);
            q.memcpy(p, hp.data(), pb); q.memcpy(s, hs.data(), sb); q.memcpy(z, hz.data(), zb);
            allocs.push_back(p); allocs.push_back(s); allocs.push_back(z);
            QuantWeight& w = ws[c];
            w.fmt = Fmt::INT4; w.N = N; w.K = K; w.payload = p; w.scales = s; w.zeros = z;
            w.row_bytes = K / 2; w.row_scales = rsc;
        }
        std::vector<float> hx(size_t(4) * K);
        for (auto& v : hx) v = float(int(rng() % 2001) - 1000) / 1000.0f;
        float* x = sycl::malloc_device<float>(size_t(4) * K, q);
        sycl_bf16* xb = sycl::malloc_device<sycl_bf16>(size_t(4) * K, q);
        float* y = sycl::malloc_device<float>(size_t(4) * N, q);
        float* yr = sycl::malloc_device<float>(size_t(4) * N, q);
        std::vector<sycl_bf16> hxb(hx.size());
        for (size_t i = 0; i < hx.size(); ++i) hxb[i] = sycl_bf16(hx[i]);
        q.memcpy(x, hx.data(), hx.size() * 4); q.memcpy(xb, hxb.data(), hxb.size() * 2); q.wait();
        const int kp = kp_for(K);
        std::printf("\n%-8s N=%-6d K=%-5d  %.1f MB/copy x%d  KP=%d KS=%d\n", sh.nm, N, K, wb / 1e6, copies, kp, K / kp);
        auto timeit = [&](const char* nm, const std::function<void(const QuantWeight&)>& f) {
            for (int i = 0; i < 2 * copies; ++i) f(ws[i % copies]);
            q.wait();
            const int it = std::max(20, 4 * copies);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i) f(ws[i % copies]);
            q.wait();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
            std::printf("  %-26s %8.1f us  %6.1f GB/s\n", nm, us, wb / us / 1e3);
        };
        auto check = [&](const char* nm, int M) {
            std::vector<float> a(size_t(M) * N), b(size_t(M) * N);
            q.memcpy(a.data(), y, a.size() * 4); q.memcpy(b.data(), yr, b.size() * 4); q.wait();
            double mx = 0, md = 0;
            for (size_t i = 0; i < a.size(); ++i) { mx = std::max(mx, double(std::fabs(b[i]))); md = std::max(md, double(std::fabs(a[i] - b[i]))); }
            std::printf("  check %-20s rel %.2e\n", nm, mx > 0 ? md / mx : md);
        };
        // reference: decode GEMV row by row
        for (int m = 0; m < 4; ++m) launch_gemv(q, ws[0], x + size_t(m) * K, yr + size_t(m) * N);
        q.wait();
        timeit("gemv M=1 (engine)", [&](const QuantWeight& w) { launch_gemv(q, w, x, y); });
        if (std::getenv("ROWS")) timeit("gemv x4 rows (engine)", [&](const QuantWeight& w) {
            for (int m = 0; m < 4; ++m) launch_gemv(q, w, x + size_t(m) * K, y + size_t(m) * N); });
        if (int4_smallm_ok(ws[0], 4, xb, y)) {
            launch_int4_smallm(q, ws[0], xb, y, 4); check("dpas smallm M=4", 4);
            timeit("dpas smallm M=4 (engine)", [&](const QuantWeight& w) { launch_int4_smallm(q, w, xb, y, 4); });
            launch_int4_smallm_wide(q, ws[0], xb, y, 4); check("dpas wide (256 GRF) M=4", 4);
            timeit("dpas wide (256 GRF) M=4", [&](const QuantWeight& w) { launch_int4_smallm_wide(q, w, xb, y, 4); });
            timeit("dpas smallm M=8 (engine)", [&](const QuantWeight& w) { launch_int4_smallm(q, w, xb, y, 8); });
            timeit("dpas wide (256 GRF) M=8", [&](const QuantWeight& w) { launch_int4_smallm_wide(q, w, xb, y, 8); });
            timeit("dpas smallm M=2 (engine)", [&](const QuantWeight& w) { launch_int4_smallm(q, w, xb, y, 2); });
        }
        {
            std::vector<int> kss = {0, -2, -3, -4, -5, -6, -8, -10, -12, -14, -16, -20, -24, -28, -34, -40};
            int last = -1;
            for (int ksm : kss) {
                const SmallmPlan pl = plan_x(N, K, ksm);
                if (pl.KS == last && ksm != 0) continue;
                if (ksm != 0) last = pl.KS;
                char nm[64];
                std::snprintf(nm, sizeof nm, "dpas %s KS=%d kc=%d TPT=%d", ksm == 0 ? "RULE" : "local", pl.KS, pl.kc, pl.TPT);
                if (ksm == 0) { sm_local<1, 128>(q, ws[0], xb, y, 4, ksm); check(nm, 4); }
                timeit(nm, [&](const QuantWeight& w) { sm_local<1, 128>(q, w, xb, y, 4, ksm); });
            }
        }
        if (std::getenv("ROWS")) {
        rows_k<8, 128, 4, 0>(q, ws[0], x, y, 4, kp); check("local R8 M4", 4);
        timeit("local R8 MB4 full", [&](const QuantWeight& w) { rows_k<8, 128, 4, 0>(q, w, x, y, 4, kp); });
        timeit("local R8 MB4 loads only", [&](const QuantWeight& w) { rows_k<8, 128, 4, 1>(q, w, x, y, 4, kp); });
        timeit("local R8 MB4 decode only", [&](const QuantWeight& w) { rows_k<8, 128, 4, 2>(q, w, x, y, 4, kp); });
        timeit("local R8 MB1 full", [&](const QuantWeight& w) { rows_k<8, 128, 1, 0>(q, w, x, y, 1, kp); });
        timeit("local R4 MB4 full", [&](const QuantWeight& w) { rows_k<4, 128, 4, 0>(q, w, x, y, 4, kp); });
        timeit("local R2 MB4 full", [&](const QuantWeight& w) { rows_k<2, 128, 4, 0>(q, w, x, y, 4, kp); });
        }
        for (void* p : allocs) sycl::free(p, q);
        sycl::free(x, q); sycl::free(xb, q); sycl::free(y, q); sycl::free(yr, q);
    }
    return 0;
}
