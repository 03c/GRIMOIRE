// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

// moe_smallm_probe.cpp -- the grouped small-M MXFP4 MoE kernels of batched
// decode (gemm_fast.cpp moe_mxfp4_smallm_impl) on Ornith's shapes: M tokens
// x top-8 of 256 experts, gate_up [256*1024][2048] (SwiGLU epilogue, bf16 h)
// and down [256*2048][512] (fp32 out).  Weights in several full copies
// (> 1 GB between reuses) so every timed launch streams from DRAM; a new
// random routing per launch.  Effective GB/s = bytes of the TOUCHED experts
// / time.  Correctness against a CPU reference on the first routing.
//
// Build (from the repo root, grimoire-dev):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 \
//     -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -cl-intel-256-GRF-per-thread" \
//     -O3 -std=c++20 tools/moe_smallm_probe.cpp -o bin/moe_smallm_probe
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <cstdlib>
#include <set>
#include <vector>

namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

static float e2m1_f(int n) {
    static const float t[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = t[n & 7];
    return (n & 8) ? -v : v;
}

struct Cfg { int KS_max; int PFD; bool sc_up; };

// VAR knobs: PFD = prefetch distance in 128-K steps (1 = engine kernel);
// SCUP = all of the thread's scales in one transposed 2-D load up front.
template <int EPI, int RBN, int PFD, bool SCUP>
sycl::event moe_kernel(sycl::queue& q, const uint8_t* pay, const uint8_t* scl, int Wrows, int K,
                       int Ne, const bf16* A, void* out, const int32_t* tile_e,
                       const int32_t* off, const int32_t* cnt, int T, int KSmax) {
    constexpr int CT = 16;
    constexpr int NCB = EPI == 1 ? 2 : 1;
    constexpr int ACC = NCB * RBN * 128;
    constexpr int WGMAX = 32 / (NCB * RBN) < 8 ? 8 : 32 / (NCB * RBN);
    const int cols = EPI == 1 ? Ne / 2 : Ne;
    const int ctiles = cols / CT;
    int KS = 1;
    while (KS < KSmax && T * ctiles * KS < 4096 && K / (KS * 2) >= 128 && KS * 2 <= WGMAX) KS *= 2;
    int kc = (K + KS - 1) / KS;
    kc = (kc + 127) / 128 * 128;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= WGMAX && TPT * 2 <= ctiles) TPT *= 2;
    const int CG = (ctiles + TPT - 1) / TPT;
    const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
    const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
    const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(Wrows) - 1, PP = unsigned(K) / 2 - 1;
    const unsigned SW = unsigned(K) / 32 - 1, SP = unsigned(K) / 32 - 1;
    const unsigned XW = unsigned(K) * 2 - 1;
    return q.parallel_for(sycl::nd_range<1>(size_t(T) * CG * TPT * KS, size_t(TPT) * KS),
                          [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<WGMAX * ACC * 4>();
        const int lid = int(it.get_local_id(0));
        const int tw = lid / KS, ks = lid % KS;
        const int g = int(it.get_group(0));
        const int t = g / CG, ct = (g % CG) * TPT + tw;
        const int e = tile_e[t];
        if (e < 0) return;
        const int rows = cnt[e];
        if (rows <= 0) return;
        const bool live = ct < ctiles;
        const int j0 = (live ? ct : 0) * CT;
        const int nrb = (rows + 7) / 8;
        const bf16* Ae = A + size_t(off[e]) * K;
        const unsigned XH = unsigned(rows) - 1;
        int wr[NCB];
        wr[0] = e * Ne + j0;
        if constexpr (NCB == 2) wr[1] = e * Ne + cols + j0;
        const int kb = ks * kc;
        const int ke = kb + kc < K ? kb + kc : K;
        constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
        es::simd<float, ACC> acc = 0.0f;
        es::simd<uint32_t, 8 * CT> swall[NCB];   // SCUP: [8 dwords = 1024 K][16 rows]
        if (live) {
            if constexpr (PFD > 1 && PFD != 9) {
                #pragma unroll
                for (int cb = 0; cb < NCB; ++cb)
                    #pragma unroll
                    for (int d = 1; d < PFD; ++d)
                        if (kb + 128 * d < ke)
                            es::prefetch_2d<uint32_t, 16, CT>(payw, PW, PH, PP, (kb + 128 * d) / 8, wr[cb], PFH);
            }
            if constexpr (SCUP) {
                #pragma unroll
                for (int cb = 0; cb < NCB; ++cb)
                    swall[cb] = es::load_2d<uint32_t, 8, CT, 1, true, false>(sclw, SW, PH, SP, kb / 128, wr[cb]);
            }
            for (int k = kb; k < ke; k += 128) {
                es::simd<uint32_t, 8 * CT> tw0[NCB], tw1[NCB];
                es::simd<uint32_t, CT> sw[NCB];
                #pragma unroll
                for (int cb = 0; cb < NCB; ++cb) {
                    if (k + 128 * (PFD == 9 ? 1 : PFD) < ke)
                        es::prefetch_2d<uint32_t, 16, CT>(payw, PW, PH, PP, (k + 128 * (PFD == 9 ? 1 : PFD)) / 8, wr[cb], PFH);
                    tw0[cb] = es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8, wr[cb]);
                    tw1[cb] = es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, PW, PH, PP, k / 8 + 8, wr[cb]);
                    if constexpr (SCUP) sw[cb] = swall[cb].template select<CT, 1>(((k - kb) / 128) * CT);
                    else sw[cb] = es::load_2d<uint32_t, 1, CT>(sclw, SW, PH, SP, k / 128, wr[cb]);
                }
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const int kk = k + 32 * b;
                    es::simd<bf16, 128> a0[RBN], a1[RBN];
                    #pragma unroll
                    for (int rb = 0; rb < RBN; ++rb) {
                        if (rb < nrb) {
                            a0[rb] = es::load_2d<bf16, 16, 8>(Ae, XW, XH, XW, kk, rb * 8);
                            a1[rb] = es::load_2d<bf16, 16, 8>(Ae, XW, XH, XW, kk + 16, rb * 8);
                        }
                    }
                    #pragma unroll
                    for (int cb = 0; cb < NCB; ++cb) {
                        es::simd<uint32_t, 16> ev = (sw[cb] >> (8 * b)) & 0xFFu;
                        es::simd<uint32_t, 16> sbits = (ev + 14u) << 23;
                        es::simd<float, 16> sc = sbits.template bit_cast_view<float>().read();
                        es::simd<uint32_t, 128> vb[2];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int dw = 4 * b + j;
                            es::simd<uint32_t, 16> wv;
                            if (dw < 8) wv = tw0[cb].template select<16, 1>(dw * CT);
                            else        wv = tw1[cb].template select<16, 1>((dw - 8) * CT);
                            if constexpr (PFD == 9) {      // timing only: no dequant
                                #pragma unroll
                                for (int qb = 0; qb < 4; ++qb)
                                    vb[(4 * j + qb) >> 3].template select<16, 1>(((4 * j + qb) & 7) * 16) = wv ^ (qb * 0x01010101u);
                                continue;
                            }
                            #pragma unroll
                            for (int qb = 0; qb < 4; ++qb) {
                                es::simd<uint32_t, 16> tb = qb ? (wv >> (8 * qb)) : wv;
                                es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                                es::simd<uint32_t, 16> hb = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                                es::simd<sycl::half, 32> hv = hb.template bit_cast_view<sycl::half>().read();
                                es::simd<float, 32> fv = hv;
                                es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                                es::simd<uint16_t, 32> bb = fb >> 16;
                                const int kp = 4 * j + qb;
                                vb[kp >> 3].template select<16, 1>((kp & 7) * 16) =
                                    bb.template bit_cast_view<uint32_t>().read();
                            }
                        }
                        #pragma unroll
                        for (int rb = 0; rb < RBN; ++rb) {
                            if (rb < nrb) {
                                es::simd<float, 128> tmp = 0.0f;
                                tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(
                                    tmp, vb[0].template bit_cast_view<bf16>().read(), a0[rb]);
                                tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(
                                    tmp, vb[1].template bit_cast_view<bf16>().read(), a1[rb]);
                                #pragma unroll
                                for (int r = 0; r < 8; ++r)
                                    acc.template select<16, 1>((rb * NCB + cb) * 128 + 16 * r) +=
                                        tmp.template select<16, 1>(16 * r) * sc;
                            }
                        }
                    }
                }
            }
        }
        if (KS > 1) {
            es::slm_block_store<float, ACC>(lid * ACC * 4, acc);
            es::barrier();
            if (ks != 0 || !live) return;
            for (int i = 1; i < KS; ++i)
                acc += es::slm_block_load<float, ACC>((tw * KS + i) * ACC * 4);
        } else if (!live) {
            return;
        }
        if constexpr (EPI == 0) {
            float* Oe = static_cast<float*>(out) + size_t(off[e]) * Ne;
            const unsigned OW = unsigned(Ne) * 4 - 1;
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb)
                if (rb < nrb)
                    es::store_2d<float, 16, 8>(Oe, OW, XH, OW, j0, rb * 8,
                        es::simd<float, 128>(acc.template select<128, 1>(rb * 128)));
        } else {
            uint32_t* He = reinterpret_cast<uint32_t*>(static_cast<bf16*>(out) + size_t(off[e]) * cols);
            const unsigned OW = unsigned(cols) * 2 - 1;
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb)
                if (rb < nrb) {
                    es::simd<float, 128> gt = acc.template select<128, 1>((rb * 2 + 0) * 128);
                    es::simd<float, 128> up = acc.template select<128, 1>((rb * 2 + 1) * 128);
                    es::simd<float, 128> hv = gt * es::inv(1.0f + es::exp2(gt * -1.4426950408889634f)) * up;
                    es::simd<uint32_t, 128> u = hv.template bit_cast_view<uint32_t>().read();
                    es::simd<uint32_t, 128> r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
                    es::simd<uint32_t, 64> lo = r.template select<64, 2>(0), hi = r.template select<64, 2>(1);
                    es::simd<uint32_t, 64> pk = lo | (hi << 16);
                    es::store_2d<uint32_t, 8, 8>(He, OW, XH, OW, j0 / 2, rb * 8, pk);
                }
        }
    });
}

// "AW" design: weights are the DPAS A operand (row-major, 8 output features x
// 16 K per DPAS), so each thread reads its weight rows as 64-byte-wide 2-D
// tiles (8 rows x 128 K per load); activations are the B operand, loaded
// with transposed 2-D loads as VNNI [8 k-pairs][16 tokens] (tokens >= rows
// read as zero).  C = [8 features][16 tokens].  RBW row-blocks of 8 features
// per thread (EPI 1: RBW gate blocks + the matching RBW up blocks).
template <int EPI, int RBW, bool LUT = false>
sycl::event moe_kernel_aw(sycl::queue& q, const uint8_t* pay, const uint8_t* scl, int Wrows, int K,
                          int Ne, const bf16* A, void* out, const int32_t* tile_e,
                          const int32_t* off, const int32_t* cnt, int T, int KSmax,
                          const uint32_t* lut = nullptr) {
    constexpr int NB = EPI == 1 ? 2 * RBW : RBW;        // row-blocks of 8 weight rows
    constexpr int ACC = NB * 128;
    constexpr int WGMAX = 16;
    const int cols = EPI == 1 ? Ne / 2 : Ne;
    const int ctiles = cols / (8 * RBW);
    int KS = 1;
    while (KS < KSmax && T * ctiles * KS < 4096 && K / (KS * 2) >= 128 && KS * 2 <= WGMAX) KS *= 2;
    int kc = (K + KS - 1) / KS;
    kc = (kc + 127) / 128 * 128;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= WGMAX && TPT * 2 <= ctiles) TPT *= 2;
    const int CG = (ctiles + TPT - 1) / TPT;
    const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
    const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
    const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(Wrows) - 1, PP = unsigned(K) / 2 - 1;
    const unsigned SW = unsigned(K) / 32 - 1, SP = unsigned(K) / 32 - 1;
    const unsigned XW = unsigned(K) * 2 - 1;
    return q.parallel_for(sycl::nd_range<1>(size_t(T) * CG * TPT * KS, size_t(TPT) * KS),
                          [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        constexpr unsigned LUTOFF = WGMAX * ACC * 4;
        es::slm_init<WGMAX * ACC * 4 + (LUT ? 1024 : 0)>();
        const int lid = int(it.get_local_id(0));
        const int tw = lid / KS, ks = lid % KS;
        const int g0 = int(it.get_group(0));
        const int t = g0 / CG, ct = (g0 % CG) * TPT + tw;
        const int e = tile_e[t];
        if (e < 0) return;
        const int rows = cnt[e];
        if (rows <= 0) return;
        if constexpr (LUT) {
            for (int i = lid; i < 8; i += TPT * KS)
                es::slm_block_store<uint32_t, 32>(LUTOFF + i * 128, es::block_load<uint32_t, 32>(lut + i * 32));
            es::barrier();
        }
        const bool live = ct < ctiles;
        const int c0 = (live ? ct : 0) * 8 * RBW;
        const uint32_t* Aw = reinterpret_cast<const uint32_t*>(A + size_t(off[e]) * K);
        const unsigned XH = unsigned(rows) - 1;
        int wr[NB];
        #pragma unroll
        for (int b = 0; b < NB; ++b)
            wr[b] = e * Ne + (EPI == 1 && b >= RBW ? cols : 0) + c0 + 8 * (b % RBW);
        const int kb = ks * kc;
        const int ke = kb + kc < K ? kb + kc : K;
        constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
        const es::simd<uint32_t, 16> shv = (es::simd<uint32_t, 16>(0, 1) & 3u) << 3;
        es::simd<float, ACC> acc = 0.0f;
        if (live) {
            for (int k = kb; k < ke; k += 128) {
                es::simd<uint32_t, 128> wt[NB];       // [8 rows][16 dwords = 128 K]
                es::simd<uint32_t, 8> sw[NB];         // [8 rows] 4 E8M0 bytes = 128 K
                #pragma unroll
                for (int b = 0; b < NB; ++b) {
                    if (k + 128 < ke)
                        es::prefetch_2d<uint32_t, 16, 8>(payw, PW, PH, PP, (k + 128) / 8, wr[b], PFH);
                    wt[b] = es::load_2d<uint32_t, 16, 8>(payw, PW, PH, PP, k / 8, wr[b]);
                    sw[b] = es::load_2d<uint32_t, 1, 8>(sclw, SW, PH, SP, k / 128, wr[b]);
                }
                #pragma unroll
                for (int g = 0; g < 4; ++g) {             // 32-K blocks
                    es::simd<uint32_t, 128> bt0 = es::load_2d<uint32_t, 8, 16, 1, true, false>(Aw, XW, XH, XW, (k + 32 * g) / 2, 0);
                    es::simd<uint32_t, 128> bt1 = es::load_2d<uint32_t, 8, 16, 1, true, false>(Aw, XW, XH, XW, (k + 32 * g) / 2 + 8, 0);
                    #pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        es::simd<uint32_t, 64> a0, a1;       // A tiles [8 rows][8 dwords]
                        #pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            es::simd<uint32_t, 16> rep = wt[b].template replicate_vs_w_hs<4, 1, 4, 0>(r * 16 + 4 * g);
                            es::simd<uint32_t, 16> pr;
                            if constexpr (LUT) {
                                es::simd<uint32_t, 16> addr = (((rep >> shv) & 0xFFu) << 2) + LUTOFF;
                                pr = es::slm_gather<uint32_t, 16>(addr);
                            } else {
                            es::simd<uint32_t, 16> tb = rep >> shv;
                            es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                            es::simd<uint32_t, 16> hb = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                            es::simd<sycl::half, 32> hv = hb.template bit_cast_view<sycl::half>().read();
                            es::simd<float, 32> fv = hv;
                            es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                            es::simd<uint16_t, 32> bb = fb >> 16;
                            pr = bb.template bit_cast_view<uint32_t>().read();
                            }
                            a0.template select<8, 1>(r * 8) = pr.template select<8, 1>(0);
                            a1.template select<8, 1>(r * 8) = pr.template select<8, 1>(8);
                        }
                        es::simd<float, 128> tmp = 0.0f;
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(
                            tmp, bt0.template bit_cast_view<bf16>().read(), a0.template bit_cast_view<bf16>().read());
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(
                            tmp, bt1.template bit_cast_view<bf16>().read(), a1.template bit_cast_view<bf16>().read());
                        es::simd<uint32_t, 8> ev = (sw[b] >> (8 * g)) & 0xFFu;
                        es::simd<uint32_t, 8> sbits = LUT ? (ev << 23) : ((ev + 14u) << 23);
                        es::simd<float, 8> sc8 = sbits.template bit_cast_view<float>().read();
                        #pragma unroll
                        for (int r = 0; r < 8; ++r)
                            acc.template select<16, 1>(b * 128 + 16 * r) += tmp.template select<16, 1>(16 * r) * sc8[r];
                    }
                }
            }
        }
        if (KS > 1) {
            es::slm_block_store<float, ACC>(lid * ACC * 4, acc);
            es::barrier();
            if (ks != 0 || !live) return;
            for (int i = 1; i < KS; ++i)
                acc += es::slm_block_load<float, ACC>((tw * KS + i) * ACC * 4);
        } else if (!live) {
            return;
        }
        const int nr = rows < 16 ? rows : 16;
        if constexpr (EPI == 0) {
            float* Oe = static_cast<float*>(out) + size_t(off[e]) * Ne;
            for (int n = 0; n < nr; ++n) {
                #pragma unroll
                for (int b = 0; b < NB; ++b) {
                    es::simd<float, 8> col = acc.template select<8, 16>(b * 128 + n);
                    es::block_store<float, 8>(Oe + size_t(n) * Ne + c0 + 8 * b, col);
                }
            }
        } else {
            uint16_t* He = reinterpret_cast<uint16_t*>(static_cast<bf16*>(out) + size_t(off[e]) * cols);
            for (int n = 0; n < nr; ++n) {
                #pragma unroll
                for (int b = 0; b < RBW; ++b) {
                    es::simd<float, 8> gt = acc.template select<8, 16>(b * 128 + n);
                    es::simd<float, 8> up = acc.template select<8, 16>((RBW + b) * 128 + n);
                    es::simd<float, 8> hv = gt * es::inv(1.0f + es::exp2(gt * -1.4426950408889634f)) * up;
                    es::simd<uint32_t, 8> u = hv.template bit_cast_view<uint32_t>().read();
                    es::simd<uint32_t, 8> r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
                    es::simd<uint32_t, 4> pk = r.template select<4, 2>(0) | (r.template select<4, 2>(1) << 16);
                    es::block_store<uint32_t, 4>(reinterpret_cast<uint32_t*>(He + size_t(n) * cols + c0 + 8 * b), pk);
                }
            }
        }
    });
}

struct Route { std::vector<int32_t> te, off, cnt; int T; int distinct; };

static Route make_route(std::mt19937& rng, int M, int E, int topk) {
    std::vector<int> c(E, 0);
    for (int m = 0; m < M; ++m) {
        std::set<int> s;
        while (int(s.size()) < topk) s.insert(int(rng() % E));
        for (int x : s) ++c[x];
    }
    // PROBE_CONTIG=1: same rows-per-expert counts, but the touched experts
    // moved to one contiguous id range (TLB / DRAM locality test)
    static const bool contig = std::getenv("PROBE_CONTIG") != nullptr;
    if (contig) {
        std::vector<int> touched;
        for (int e = 0; e < E; ++e) if (c[e]) touched.push_back(c[e]);
        const int D = int(touched.size());
        const int base = int(rng() % (E - D + 1));
        std::fill(c.begin(), c.end(), 0);
        for (int i = 0; i < D; ++i) c[base + i] = touched[i];
    }
    Route r; r.off.assign(E, 0); r.cnt.assign(E, 0);
    int o = 0;
    for (int e = 0; e < E; ++e) { r.off[e] = o; r.cnt[e] = c[e]; o += c[e]; if (c[e]) r.te.push_back(e); }
    r.distinct = int(r.te.size());
    r.T = M * topk;                          // the engine's host-known cap
    while (int(r.te.size()) < r.T) r.te.push_back(-1);
    return r;
}

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int iters = argc > 1 ? std::atoi(argv[1]) : 50;
    const int E = 256, H = 2048, I = 512, TOPK = 8;
    const int NC = 3;                                   // full weight copies
    std::mt19937 rng(7);
    struct Mat { int N, K; std::vector<uint8_t> hp, hs; std::vector<uint8_t*> dp, ds; };
    Mat mats[2] = {{E * 2 * I, H, {}, {}, {}, {}}, {E * H, I, {}, {}, {}, {}}};
    for (auto& m : mats) {
        const size_t pb = size_t(m.N) * m.K / 2, sb = size_t(m.N) * m.K / 32;
        m.hp.resize(pb); m.hs.resize(sb);
        for (auto& x : m.hp) x = uint8_t(rng());
        for (auto& x : m.hs) x = uint8_t(118 + rng() % 12);
        for (int c = 0; c < NC; ++c) {
            m.dp.push_back(sycl::malloc_device<uint8_t>(pb, q));
            m.ds.push_back(sycl::malloc_device<uint8_t>(sb, q));
            q.memcpy(m.dp[c], m.hp.data(), pb); q.memcpy(m.ds[c], m.hs.data(), sb);
        }
    }
    // byte -> (bf16(e2m1(hi)) << 16) | bf16(e2m1(lo)), the unscaled values (exact in bf16)
    std::vector<uint32_t> hlut(256);
    for (int b = 0; b < 256; ++b) {
        auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u >> 16; };
        hlut[b] = bits(e2m1_f(b & 15)) | (bits(e2m1_f(b >> 4)) << 16);
    }
    uint32_t* dlut = sycl::malloc_device<uint32_t>(256, q);
    q.memcpy(dlut, hlut.data(), 1024);
    q.wait();
    for (int M : {2, 4, 8, 16}) {
        const int R = M * TOPK;
        constexpr int NR = 16;
        std::vector<Route> routes;
        std::vector<int32_t*> dte(NR), doff(NR), dcnt(NR);
        for (int i = 0; i < NR; ++i) {
            routes.push_back(make_route(rng, M, E, TOPK));
            dte[i] = sycl::malloc_device<int32_t>(R, q);
            doff[i] = sycl::malloc_device<int32_t>(E, q);
            dcnt[i] = sycl::malloc_device<int32_t>(E, q);
            q.memcpy(dte[i], routes[i].te.data(), R * 4);
            q.memcpy(doff[i], routes[i].off.data(), E * 4);
            q.memcpy(dcnt[i], routes[i].cnt.data(), E * 4);
        }
        std::vector<bf16> hx(size_t(R) * H), hh(size_t(R) * I);
        std::uniform_real_distribution<float> U(-1.f, 1.f);
        for (auto& v : hx) v = bf16(U(rng));
        for (auto& v : hh) v = bf16(U(rng));
        bf16* dx = sycl::malloc_device<bf16>(hx.size(), q);
        bf16* dhin = sycl::malloc_device<bf16>(hh.size(), q);      // down input (fixed)
        bf16* dh = sycl::malloc_device<bf16>(size_t(R) * I, q);    // gate_up output
        float* dy = sycl::malloc_device<float>(size_t(R) * H, q);
        q.memcpy(dx, hx.data(), hx.size() * 2); q.memcpy(dhin, hh.data(), hh.size() * 2); q.wait();
        double avg_distinct = 0;
        for (auto& r : routes) avg_distinct += r.distinct;
        avg_distinct /= NR;

        struct V { const char* name; int which; };
        for (int which = 0; which < 2; ++which) {           // 0 gate_up, 1 down
            const Mat& mt = mats[which];
            const double ebytes = double(which == 0 ? 2 * I : H) * mt.K * (0.5 + 1.0 / 32);
            for (int var = 0; var < 10; ++var) {
                int ci = 0;
                auto launch = [&](int ri) {
                    const uint8_t* p = mt.dp[ci % NC]; const uint8_t* s = mt.ds[ci % NC]; ++ci;
                    const int ks = var == 5 ? 1 : 16;
                    if (var == 6) return which == 0 ? moe_kernel_aw<1, 1>(q, p, s, mt.N, mt.K, 2 * I, dx, dh, dte[ri], doff[ri], dcnt[ri], R, ks)
                                                    : moe_kernel_aw<0, 1>(q, p, s, mt.N, mt.K, H, dhin, dy, dte[ri], doff[ri], dcnt[ri], R, ks);
                    if (var == 8) return which == 0 ? moe_kernel_aw<1, 1, true>(q, p, s, mt.N, mt.K, 2 * I, dx, dh, dte[ri], doff[ri], dcnt[ri], R, ks, dlut)
                                                    : moe_kernel_aw<0, 1, true>(q, p, s, mt.N, mt.K, H, dhin, dy, dte[ri], doff[ri], dcnt[ri], R, ks, dlut);
                    if (var == 9) return which == 0 ? moe_kernel_aw<1, 2, true>(q, p, s, mt.N, mt.K, 2 * I, dx, dh, dte[ri], doff[ri], dcnt[ri], R, ks, dlut)
                                                    : moe_kernel_aw<0, 2, true>(q, p, s, mt.N, mt.K, H, dhin, dy, dte[ri], doff[ri], dcnt[ri], R, ks, dlut);
                    if (var == 7) return which == 0 ? moe_kernel_aw<1, 2>(q, p, s, mt.N, mt.K, 2 * I, dx, dh, dte[ri], doff[ri], dcnt[ri], R, ks)
                                                    : moe_kernel_aw<0, 2>(q, p, s, mt.N, mt.K, H, dhin, dy, dte[ri], doff[ri], dcnt[ri], R, ks);
                    if (which == 0) {
                        if (M <= 8) {
                            if (var == 0) return moe_kernel<1, 1, 1, false>(q, p, s, mt.N, mt.K, 2 * I, dx, dh, dte[ri], doff[ri], dcnt[ri], R, ks);
                            if (var == 1) return moe_kernel<1, 1, 2, false>(q, p, s, mt.N, mt.K, 2 * I, dx, dh, dte[ri], doff[ri], dcnt[ri], R, ks);
                            if (var == 2) return moe_kernel<1, 1, 9, false>(q, p, s, mt.N, mt.K, 2 * I, dx, dh, dte[ri], doff[ri], dcnt[ri], R, ks);
                            if (var == 3) return moe_kernel<1, 1, 1, true>(q, p, s, mt.N, mt.K, 2 * I, dx, dh, dte[ri], doff[ri], dcnt[ri], R, ks);
                            if (var == 4) return moe_kernel<1, 1, 2, true>(q, p, s, mt.N, mt.K, 2 * I, dx, dh, dte[ri], doff[ri], dcnt[ri], R, ks);
                            return moe_kernel<1, 1, 2, false>(q, p, s, mt.N, mt.K, 2 * I, dx, dh, dte[ri], doff[ri], dcnt[ri], R, ks);
                        }
                        if (var == 0) return moe_kernel<1, 2, 1, false>(q, p, s, mt.N, mt.K, 2 * I, dx, dh, dte[ri], doff[ri], dcnt[ri], R, ks);
                        return moe_kernel<1, 2, 2, true>(q, p, s, mt.N, mt.K, 2 * I, dx, dh, dte[ri], doff[ri], dcnt[ri], R, ks);
                    }
                    if (M <= 8) {
                        if (var == 0) return moe_kernel<0, 1, 1, false>(q, p, s, mt.N, mt.K, H, dhin, dy, dte[ri], doff[ri], dcnt[ri], R, ks);
                        if (var == 1) return moe_kernel<0, 1, 2, false>(q, p, s, mt.N, mt.K, H, dhin, dy, dte[ri], doff[ri], dcnt[ri], R, ks);
                        if (var == 2) return moe_kernel<0, 1, 9, false>(q, p, s, mt.N, mt.K, H, dhin, dy, dte[ri], doff[ri], dcnt[ri], R, ks);
                        if (var == 3) return moe_kernel<0, 1, 1, true>(q, p, s, mt.N, mt.K, H, dhin, dy, dte[ri], doff[ri], dcnt[ri], R, ks);
                        if (var == 4) return moe_kernel<0, 1, 3, true>(q, p, s, mt.N, mt.K, H, dhin, dy, dte[ri], doff[ri], dcnt[ri], R, ks);
                        return moe_kernel<0, 1, 3, true>(q, p, s, mt.N, mt.K, H, dhin, dy, dte[ri], doff[ri], dcnt[ri], R, ks);
                    }
                    if (var == 0) return moe_kernel<0, 2, 1, false>(q, p, s, mt.N, mt.K, H, dhin, dy, dte[ri], doff[ri], dcnt[ri], R, ks);
                    return moe_kernel<0, 2, 3, true>(q, p, s, mt.N, mt.K, H, dhin, dy, dte[ri], doff[ri], dcnt[ri], R, ks);
                };
                if (M > 8 && var > 1 && var < 6) continue;
                if (var >= 1 && var <= 5 && M != 8) continue;
                // correctness on routing 0, copy 0
                ci = 0; launch(0).wait();
                double maxrel = 0;
                {
                    const Route& r0 = routes[0];
                    std::vector<float> got;
                    if (which == 0) {
                        std::vector<bf16> g(size_t(R) * I); q.memcpy(g.data(), dh, g.size() * 2).wait();
                        got.resize(g.size()); for (size_t i = 0; i < g.size(); ++i) got[i] = float(g[i]);
                    } else { got.resize(size_t(R) * H); q.memcpy(got.data(), dy, got.size() * 4).wait(); }
                    int checked = 0;
                    for (int ti = 0; ti < r0.distinct && checked < 6; ++ti, ++checked) {
                        const int e = r0.te[ti];
                        for (int rr = 0; rr < r0.cnt[e]; ++rr) {
                            const int row = r0.off[e] + rr;
                            const int ncol = which == 0 ? I : H;
                            for (int n = 0; n < ncol; n += 7) {
                                auto dot = [&](int wrow) {
                                    double s = 0, mag = 0;
                                    for (int k = 0; k < mt.K; ++k) {
                                        const uint8_t byte = mt.hp[size_t(wrow) * (mt.K / 2) + k / 2];
                                        const int nib = (k & 1) ? (byte >> 4) : (byte & 15);
                                        const double w = e2m1_f(nib) * std::ldexp(1.0, int(mt.hs[size_t(wrow) * (mt.K / 32) + k / 32]) - 127);
                                        const double xv = which == 0 ? double(float(hx[size_t(row) * H + k]))
                                                                     : double(float(hh[size_t(row) * I + k]));
                                        s += w * xv; mag += std::fabs(w * xv);
                                    }
                                    return std::make_pair(s, mag);
                                };
                                double ref, mag;
                                if (which == 0) {
                                    auto [gs, gm] = dot(e * 2 * I + n);
                                    auto [us, um] = dot(e * 2 * I + I + n);
                                    ref = gs / (1.0 + std::exp(-gs)) * us;
                                    mag = std::fabs(ref) + 1e-3 * (gm + um);
                                } else {
                                    auto [s, m] = dot(e * H + n);
                                    ref = s; mag = m;
                                }
                                const double err = std::fabs(ref - got[size_t(row) * ncol + n]);
                                maxrel = std::max(maxrel, mag > 0 ? err / mag : err);
                            }
                        }
                    }
                }
                ci = 0;
                for (int i = 0; i < 3; ++i) launch(i % NR);
                q.wait();
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; ++i) launch(i % NR);
                q.wait();
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count() / iters;
                const double gbs = avg_distinct * ebytes / (ms * 1e6);
                static const char* names[10] = {"pf1       ", "pf2       ", "NODECODE  ", "pf1+scup  ", "pf2/3+scup", "ks1       ", "AW rbw1   ", "AW rbw2   ", "AWL rbw1  ", "AWL rbw2  "};
                std::printf("%-7s M=%2d experts~%5.1f %s %8.1f us  %6.1f GB/s  maxrel %.2e %s\n",
                            which == 0 ? "gate_up" : "down", M, avg_distinct, names[var], ms * 1000, gbs,
                            maxrel, maxrel < 5e-3 ? "ok" : "FAIL");
            }
        }
        for (int i = 0; i < NR; ++i) { sycl::free(dte[i], q); sycl::free(doff[i], q); sycl::free(dcnt[i], q); }
        sycl::free(dx, q); sycl::free(dhin, q); sycl::free(dh, q); sycl::free(dy, q);
    }
    return 0;
}
