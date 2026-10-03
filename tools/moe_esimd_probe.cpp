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

// moe_esimd_probe.cpp -- grouped MoE MXFP4 GEMM on the B70's matrix unit in
// ESIMD, standalone: correctness against a CPU reference + TFLOP/s on the
// shapes of Ornith's routed experts (256 experts, top-8, ~187 rows each on a
// 5987-token prompt; gate_up N=1024 K=2048 with the SwiGLU epilogue, down
// N=2048 K=512).  The joint_matrix grouped kernel in gemm_fast.cpp stages the
// dequantized B tile in SLM with a work-group barrier every 32 K and reaches
// ~25 TFLOP/s in-model; here every thread dequantizes its own B tile in
// registers and nothing synchronizes after the LUT is staged.
//
//   thread tile: 64 rows (8 DPAS row blocks) x 32 columns (two 16-row weight
//   halves: EPI 0 = 32 consecutive output columns, EPI 1 = 16 gate rows + the
//   16 matching up rows -> 16 SwiGLU outputs)
//   work-group: 4 m-threads x 4 n-threads over one 256-row tile of an expert
//   (the engine's tile table: tile_e / tile_mb, MC = 256 rows), so the four
//   n-threads share A and the four m-threads share B through L1.
//
//   SCALE 0: unscaled bf16 B; two DPAS per MX block into a temporary, then
//            acc += tmp * scale (fp32 FMA per column)
//   SCALE 1: B scaled at conversion (fp32 multiply by the power-of-two block
//            scale, exact) -- the same bf16 B values and the same DPAS order
//            as the joint_matrix kernel, so the results should be identical.
//
// Build (256-GRF, AOT for the B70), from the repo root:
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 \
//     -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -cl-intel-256-GRF-per-thread" \
//     -O3 -std=c++20 tools/moe_esimd_probe.cpp -o bin/moe_esimd_probe
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace es = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

static float e2m1_f(int n) {
    static const float t[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = t[n & 7];
    return (n & 8) ? -v : v;
}
static uint32_t bf16_bits_exact(float f) {   // e2m1 values are exact in bf16
    uint32_t u; std::memcpy(&u, &f, 4); return u >> 16;
}
static float bf16_round(float f) {           // RNE, finite inputs
    uint32_t u; std::memcpy(&u, &f, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    float r; std::memcpy(&r, &u, 4); return r;
}

constexpr int MC = 256;                       // rows per tile (engine: moe_grouped_rows())

template <int EPI, int SCALE, int RBN>
sycl::event moe_es(sycl::queue& q, const uint8_t* pay, const uint8_t* scl, const uint32_t* lut,
                   int Ne, int K, int Wrows, const bf16* A, void* out,
                   const int32_t* tile_e, const int32_t* tile_mb, const int32_t* off,
                   const int32_t* cnt, int T) {
    constexpr int PFA = 0, MT = 8 * RBN, TMT = MC / MT, TNT = RBN == 8 ? 4 : 2, TPW = TMT * TNT;
    const int FI = Ne / 2;
    const int nG = EPI == 1 ? FI / (16 * TNT) : Ne / (32 * TNT);
    return q.parallel_for(sycl::nd_range<1>(size_t(T) * nG * TPW, TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<1024>();
        const int lid = int(it.get_local_id(0));
        if (lid < 8)
            es::slm_block_store<uint32_t, 32>(lid * 128, es::block_load<uint32_t, 32>(lut + lid * 32));
        es::barrier();
        const int g = int(it.get_group(0));
        const int t = g / nG, ng = g % nG;
        const int ti = lid / TNT, tj = lid % TNT;
        const int e = tile_e[t];
        const int M = cnt[e];
        const int m0 = tile_mb[t] * MC + ti * MT;
        if (m0 >= M) return;
        const int nb = ng * TNT + tj;
        const int rh0 = e * Ne + (EPI == 1 ? nb * 16 : nb * 32);
        const int rh1 = e * Ne + (EPI == 1 ? FI + nb * 16 : nb * 32 + 16);
        const bf16* Ae = A + size_t(off[e]) * K;
        const unsigned AW = unsigned(K) * 2 - 1, AH = unsigned(M) - 1;
        const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
        const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(Wrows) - 1;
        const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
        const unsigned KB = unsigned(K) / 32;
        constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
        const es::simd<uint32_t, 16> iv(0, 1);
        es::simd<float, RBN * 2 * 128> acc = 0.0f;
        for (int k = 0; k < K; k += 128) {
            if (k + 128 < K) {
                es::prefetch_2d<uint32_t, 16, 16>(payw, PW, PH, PW, (k + 128) / 8, rh0, PFH);
                es::prefetch_2d<uint32_t, 16, 16>(payw, PW, PH, PW, (k + 128) / 8, rh1, PFH);
            }
            // E8M0 scales of this 128-K line (byte b = MX block b) for the two halves
            es::simd<uint32_t, 16> sw0 = es::gather<uint32_t, 16>(sclw, ((iv + unsigned(rh0)) * KB + unsigned(k / 32)));
            es::simd<uint32_t, 16> sw1 = es::gather<uint32_t, 16>(sclw, ((iv + unsigned(rh1)) * KB + unsigned(k / 32)));
            #pragma unroll 1
            for (int b = 0; b < 4; ++b) {           // not unrolled: unrolling hoists all loads and spills
                const int kk = k + 32 * b;
                if (PFA > 0 && kk + 32 * PFA < K) {
                    es::prefetch_2d<bf16, 32, 32>(Ae, AW, AH, AW, kk + 32 * PFA, m0, PFH);
                    es::prefetch_2d<bf16, 32, 32>(Ae, AW, AH, AW, kk + 32 * PFA, m0 + 32, PFH);
                }
                // payload: [4 dwords][16 rows] per half (16 bytes = this MX block)
                es::simd<uint32_t, 64> tw0 = es::load_2d<uint32_t, 4, 16, 1, true, false>(payw, PW, PH, PW, kk / 8, rh0);
                es::simd<uint32_t, 64> tw1 = es::load_2d<uint32_t, 4, 16, 1, true, false>(payw, PW, PH, PW, kk / 8, rh1);
                // one 16-column half at a time: its B tiles (16 GRF) + 32 rows of A
                // (32 GRF) live next to the 128-GRF accumulator; A is re-read from L1
                // for the second half
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    es::simd<uint32_t, 16> ev;
                    if (hh == 0) ev = (sw0 >> (8 * b)) & 0xFFu;
                    else         ev = (sw1 >> (8 * b)) & 0xFFu;
                    es::simd<uint32_t, 16> sb = ev << 23;
                    const es::simd<float, 16> s16 = sb.template bit_cast_view<float>().read();
                    es::simd<uint32_t, 256> vb;              // VNNI [K half][8 k-pairs][16 cols]
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        #pragma unroll
                        for (int kp = 0; kp < 8; ++kp) {
                            const int dw = 2 * h + (kp >> 2);
                            es::simd<uint32_t, 16> wx;
                            if (hh == 0) wx = tw0.template select<16, 1>(dw * 16);
                            else         wx = tw1.template select<16, 1>(dw * 16);
                            const int j = kp & 3;
                            es::simd<uint32_t, 16> addr;
                            if (j == 0) addr = (wx << 2) & 0x3FCu;
                            else        addr = (wx >> (8 * j - 2)) & 0x3FCu;
                            es::simd<uint32_t, 16> p = es::slm_gather<uint32_t, 16>(addr);
                            if constexpr (SCALE == 1) {
                                es::simd<uint32_t, 16> lo = p << 16, hi = p & 0xFFFF0000u;
                                es::simd<float, 16> lof = lo.template bit_cast_view<float>().read() * s16;
                                es::simd<float, 16> hif = hi.template bit_cast_view<float>().read() * s16;
                                p = hif.template bit_cast_view<uint32_t>().read() |
                                    (lof.template bit_cast_view<uint32_t>().read() >> 16);
                            }
                            vb.template select<16, 1>((h * 8 + kp) * 16) = p;
                        }
                    }
                    #pragma unroll
                    for (int rg = 0; rg < RBN / 4; ++rg) {
                        // [2 K halves][32 rows][16 K] of A
                        es::simd<bf16, 1024> a = es::load_2d<bf16, 16, 32, 2>(Ae, AW, AH, AW, kk, m0 + 32 * rg);
                        #pragma unroll
                        for (int rr = 0; rr < 4; ++rr) {
                            const int rb = rg * 4 + rr;
                            const es::simd<bf16, 256> b0 = vb.template select<128, 1>(0).template bit_cast_view<bf16>().read();
                            const es::simd<bf16, 256> b1 = vb.template select<128, 1>(128).template bit_cast_view<bf16>().read();
                            const es::simd<bf16, 128> a0 = a.template select<128, 1>(rr * 128);
                            const es::simd<bf16, 128> a1 = a.template select<128, 1>(512 + rr * 128);
                            if constexpr (SCALE == 1) {
                                es::simd<float, 128> c = acc.template select<128, 1>((rb * 2 + hh) * 128);
                                c = xmx::dpas<8, 8, float, float, bf16, bf16>(c, b0, a0);
                                c = xmx::dpas<8, 8, float, float, bf16, bf16>(c, b1, a1);
                                acc.template select<128, 1>((rb * 2 + hh) * 128) = c;
                            } else {
                                es::simd<float, 128> tmp = 0.0f;
                                tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, b0, a0);
                                tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, b1, a1);
                                #pragma unroll
                                for (int r = 0; r < 8; ++r)
                                    acc.template select<16, 1>((rb * 2 + hh) * 128 + 16 * r) +=
                                        tmp.template select<16, 1>(16 * r) * s16;
                            }
                        }
                    }
                }
            }
        }
        if constexpr (EPI == 0) {
            float* Oe = static_cast<float*>(out) + size_t(off[e]) * Ne;
            const unsigned OW = unsigned(Ne) * 4 - 1;
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb)
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh)
                    es::store_2d<float, 16, 8>(Oe, OW, AH, OW, nb * 32 + 16 * hh, m0 + 8 * rb,
                        es::simd<float, 128>(acc.template select<128, 1>((rb * 2 + hh) * 128)));
        } else {
            // H bf16 [rows][FI], written as dword pairs: bf16 RNE by hand (finite values)
            uint32_t* He = reinterpret_cast<uint32_t*>(static_cast<bf16*>(out) + size_t(off[e]) * FI);
            const unsigned OW = unsigned(FI) * 2 - 1;
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb) {
                es::simd<float, 128> gt = acc.template select<128, 1>((rb * 2 + 0) * 128);
                es::simd<float, 128> up = acc.template select<128, 1>((rb * 2 + 1) * 128);
                es::simd<float, 128> hv = gt * es::inv(1.0f + es::exp2(gt * -1.4426950408889634f)) * up;
                es::simd<uint32_t, 128> u = hv.template bit_cast_view<uint32_t>().read();
                es::simd<uint32_t, 128> r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
                es::simd<uint32_t, 64> lo = r.template select<64, 2>(0), hi = r.template select<64, 2>(1);
                es::simd<uint32_t, 64> pk = lo | (hi << 16);
                es::store_2d<uint32_t, 8, 8>(He, OW, AH, OW, nb * 8, m0 + 8 * rb, pk);
            }
        }
    });
}

// v2: the work-group dequantizes each 128-K step of its B tiles ONCE into SLM
// (double-buffered, DPAS tile order: [MX block][col group][K half][8 k-pairs]
// [16 cols] dwords), one barrier per 128 K; every thread then reads its B
// tiles with 1 KB SLM block loads.  The 8 m-threads of a column group used to
// dequantize the same tile 8 times (~16 ALU ops per DPAS).  Pre-scaled (exact).
template <int EPI, int RBN>
sycl::event moe_es2(sycl::queue& q, const uint8_t* pay, const uint8_t* scl, const uint32_t* lut,
                    int Ne, int K, int Wrows, const bf16* A, void* out,
                    const int32_t* tile_e, const int32_t* tile_mb, const int32_t* off,
                    const int32_t* cnt, int T) {
    constexpr int MT = 8 * RBN, TMT = MC / MT, TNT = 2, TPW = TMT * TNT;
    constexpr int CG = 2 * TNT;                   // 16-column groups per work-group
    constexpr int STEP = 4 * CG * 2 * 512;        // bytes of dequantized B per 128 K
    const int FI = Ne / 2;
    const int nG = EPI == 1 ? FI / (16 * TNT) : Ne / (32 * TNT);
    return q.parallel_for(sycl::nd_range<1>(size_t(T) * nG * TPW, TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<1024 + 2 * STEP>();
        const int lid = int(it.get_local_id(0));
        for (int i = lid; i < 8; i += TPW)
            es::slm_block_store<uint32_t, 32>(i * 128, es::block_load<uint32_t, 32>(lut + i * 32));
        const int g = int(it.get_group(0));
        const int t = g / nG, ng = g % nG;
        const int ti = lid / TNT, tj = lid % TNT;
        const int e = tile_e[t];
        const int M = cnt[e];
        const int m0 = tile_mb[t] * MC + ti * MT;
        const bool active = m0 < M;
        const bf16* Ae = A + size_t(off[e]) * K;
        const unsigned AW = unsigned(K) * 2 - 1, AH = unsigned(M) - 1;
        const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
        const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(Wrows) - 1;
        const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
        const unsigned KB = unsigned(K) / 32;
        const es::simd<uint32_t, 16> iv(0, 1);
        // dequant role: (MX block, column group) pairs, 4 * CG per 128 K, NC per thread
        constexpr int NC = 4 * CG / TPW;
        static_assert(NC * TPW == 4 * CG, "dequant pairs must split evenly");
        es::barrier();                                // LUT staged
        es::simd<float, RBN * 2 * 128> acc = 0.0f;
        const int nsteps = K / 128;
        es::simd<uint32_t, 64 * NC> tw;
        es::simd<uint32_t, 16 * NC> sw;
        // payload + scales of step st for this thread's dequant pairs
        #define MOE_ES2_FETCH(st)                                                              \
        {                                                                                  \
            const int k1 = (st) * 128;                                                     \
            _Pragma("unroll")                                                              \
            for (int i = 0; i < NC; ++i) {                                                 \
                const int c = lid + i * TPW, mxc = c / CG, cg = c % CG;                    \
                const int tjc = cg >> 1, hhc = cg & 1, nbc = ng * TNT + tjc;               \
                const int rh = e * Ne + (EPI == 1 ? (hhc ? FI : 0) + nbc * 16 : nbc * 32 + hhc * 16); \
                tw.template select<64, 1>(64 * i) = es::load_2d<uint32_t, 4, 16, 1, true, false>( \
                    payw, PW, PH, PW, (k1 + 32 * mxc) / 8, rh);                            \
                sw.template select<16, 1>(16 * i) = es::gather<uint32_t, 16>(sclw,         \
                    (iv + unsigned(rh)) * KB + unsigned(k1 / 32));                         \
            }                                                                              \
        }
        MOE_ES2_FETCH(0)
        for (int s = -1; s < nsteps; ++s) {
            if (s >= 0 && active) {
                const unsigned base = 1024u + unsigned(s & 1) * STEP;
                #pragma unroll 1
                for (int mx = 0; mx < 4; ++mx) {
                    const int kk = s * 128 + 32 * mx;
                    #pragma unroll
                    for (int rg = 0; rg < RBN / 4; ++rg) {
                        es::simd<bf16, 1024> a = es::load_2d<bf16, 16, 32, 2>(Ae, AW, AH, AW, kk, m0 + 32 * rg);
                        #pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            es::simd<uint32_t, 256> vb = es::slm_block_load<uint32_t, 256>(
                                base + unsigned((mx * CG + tj * 2 + hh) * 2) * 512u);
                            const es::simd<bf16, 256> b0 = vb.template select<128, 1>(0).template bit_cast_view<bf16>().read();
                            const es::simd<bf16, 256> b1 = vb.template select<128, 1>(128).template bit_cast_view<bf16>().read();
                            #pragma unroll
                            for (int rr = 0; rr < 4; ++rr) {
                                const int rb = rg * 4 + rr;
                                const es::simd<bf16, 128> a0 = a.template select<128, 1>(rr * 128);
                                const es::simd<bf16, 128> a1 = a.template select<128, 1>(512 + rr * 128);
                                es::simd<float, 128> c = acc.template select<128, 1>((rb * 2 + hh) * 128);
                                c = xmx::dpas<8, 8, float, float, bf16, bf16>(c, b0, a0);
                                c = xmx::dpas<8, 8, float, float, bf16, bf16>(c, b1, a1);
                                acc.template select<128, 1>((rb * 2 + hh) * 128) = c;
                            }
                        }
                    }
                }
            }
            if (s + 1 < nsteps) {
                const unsigned base = 1024u + unsigned((s + 1) & 1) * STEP;
                #pragma unroll
                for (int i = 0; i < NC; ++i) {
                    const int cidx = lid + i * TPW, mxc = cidx / CG, cg = cidx % CG;
                    es::simd<uint32_t, 16> ev = (es::simd<uint32_t, 16>(sw.template select<16, 1>(16 * i)) >> (8 * mxc)) & 0xFFu;
                    es::simd<uint32_t, 16> sb = ev << 23;
                    sb.merge(es::simd<uint32_t, 16>(0x00400000u), ev == 0u);
                    const es::simd<float, 16> s16 = sb.template bit_cast_view<float>().read();
                    es::simd<uint32_t, 256> vb;
                    #pragma unroll
                    for (int hk = 0; hk < 2; ++hk) {
                        #pragma unroll
                        for (int kp = 0; kp < 8; ++kp) {
                            const int dw = 2 * hk + (kp >> 2);
                            es::simd<uint32_t, 16> wx = tw.template select<16, 1>(64 * i + dw * 16);
                            const int j = kp & 3;
                            es::simd<uint32_t, 16> addr;
                            if (j == 0) addr = (wx << 2) & 0x3FCu;
                            else        addr = (wx >> (8 * j - 2)) & 0x3FCu;
                            es::simd<uint32_t, 16> p = es::slm_gather<uint32_t, 16>(addr);
                            es::simd<uint32_t, 16> lo = p << 16, hi = p & 0xFFFF0000u;
                            es::simd<float, 16> lof = lo.template bit_cast_view<float>().read() * s16;
                            es::simd<float, 16> hif = hi.template bit_cast_view<float>().read() * s16;
                            vb.template select<16, 1>((hk * 8 + kp) * 16) =
                                hif.template bit_cast_view<uint32_t>().read() |
                                (lof.template bit_cast_view<uint32_t>().read() >> 16);
                        }
                    }
                    es::slm_block_store<uint32_t, 256>(base + unsigned((mxc * CG + cg) * 2) * 512u, vb);
                }
                if (s + 2 < nsteps) MOE_ES2_FETCH(s + 2)
            }
            es::barrier();
        }
        #undef MOE_ES2_FETCH
        if (!active) return;
        const int nb = ng * TNT + tj;
        if constexpr (EPI == 0) {
            float* Oe = static_cast<float*>(out) + size_t(off[e]) * Ne;
            const unsigned OW = unsigned(Ne) * 4 - 1;
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb)
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh)
                    es::store_2d<float, 16, 8>(Oe, OW, AH, OW, nb * 32 + 16 * hh, m0 + 8 * rb,
                        es::simd<float, 128>(acc.template select<128, 1>((rb * 2 + hh) * 128)));
        } else {
            uint32_t* He = reinterpret_cast<uint32_t*>(static_cast<bf16*>(out) + size_t(off[e]) * FI);
            const unsigned OW = unsigned(FI) * 2 - 1;
            #pragma unroll
            for (int rb = 0; rb < RBN; ++rb) {
                es::simd<float, 128> gt = acc.template select<128, 1>((rb * 2 + 0) * 128);
                es::simd<float, 128> up = acc.template select<128, 1>((rb * 2 + 1) * 128);
                es::simd<float, 128> hv = gt * es::inv(1.0f + es::exp2(gt * -1.4426950408889634f)) * up;
                es::simd<uint32_t, 128> u = hv.template bit_cast_view<uint32_t>().read();
                es::simd<uint32_t, 128> r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
                es::simd<uint32_t, 64> lo = r.template select<64, 2>(0), hi = r.template select<64, 2>(1);
                es::simd<uint32_t, 64> pk = lo | (hi << 16);
                es::store_2d<uint32_t, 8, 8>(He, OW, AH, OW, nb * 8, m0 + 8 * rb, pk);
            }
        }
    });
}

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int iters = argc > 1 ? std::atoi(argv[1]) : 10;
    const int E = 256, TOK = 5987, TOPK = 8, H = 2048, I = 512;
    std::mt19937 rng(7);
    // expert loads: lognormal weights, total TOK*TOPK rows (a real prompt is skewed too)
    std::vector<int> cnt(E);
    {
        std::lognormal_distribution<double> L(0.0, 0.6);
        std::vector<double> w(E); double s = 0;
        for (auto& x : w) { x = L(rng); s += x; }
        int tot = 0;
        for (int e = 0; e < E; ++e) { cnt[e] = int(TOK * TOPK * w[e] / s); tot += cnt[e]; }
        for (int e = 0; tot < TOK * TOPK; e = (e + 1) % E) { ++cnt[e]; ++tot; }
        cnt[5] = 0; cnt[77] = 1; cnt[78] = 7; cnt[200] = 300;       // edge cases
    }
    std::vector<int> off(E);
    int R = 0;
    for (int e = 0; e < E; ++e) { off[e] = R; R += cnt[e]; }
    std::vector<int> te, tmb;
    for (int e = 0; e < E; ++e)
        for (int mb = 0; mb * MC < cnt[e]; ++mb) { te.push_back(e); tmb.push_back(mb); }
    const int T = int(te.size());
    int mx = 0; for (int c : cnt) mx = std::max(mx, c);
    std::printf("rows %d, tiles %d, max rows/expert %d\n", R, T, mx);
    int32_t* dtab = sycl::malloc_device<int32_t>(2 * T + 2 * E, q);
    q.memcpy(dtab, te.data(), T * 4); q.memcpy(dtab + T, tmb.data(), T * 4);
    q.memcpy(dtab + 2 * T, off.data(), E * 4); q.memcpy(dtab + 2 * T + E, cnt.data(), E * 4).wait();
    const int32_t *d_te = dtab, *d_tmb = dtab + T, *d_off = dtab + 2 * T, *d_cnt = dtab + 2 * T + E;

    std::vector<uint32_t> hlut(256);
    for (int b = 0; b < 256; ++b)
        hlut[b] = bf16_bits_exact(e2m1_f(b & 15)) | (bf16_bits_exact(e2m1_f(b >> 4)) << 16);
    uint32_t* lut = sycl::malloc_device<uint32_t>(256, q);
    q.memcpy(lut, hlut.data(), 1024).wait();

    bool all_ok = true;
    struct Case { int epi, Ne, K; const char* what; };
    const Case cases[] = {{1, 2 * I, H, "gate_up+swiglu"}, {0, 2 * I, H, "gate_up fp32  "}, {0, H, I, "down          "}};
    for (const Case& cs : cases) {
        const int epi = cs.epi, Ne = cs.Ne, K = cs.K;
        const size_t pb = size_t(E) * Ne * K / 2, sb = size_t(E) * Ne * K / 32;
        std::vector<uint8_t> hp(pb), hs(sb);
        for (auto& x : hp) x = uint8_t(rng());
        for (auto& x : hs) x = uint8_t(118 + rng() % 12);          // 2^-9 .. 2^2
        std::vector<bf16> hx(size_t(R) * K);
        std::uniform_real_distribution<float> U(-1.f, 1.f);
        for (auto& v : hx) v = bf16(U(rng));
        uint8_t* dp = sycl::malloc_device<uint8_t>(pb, q);
        uint8_t* ds = sycl::malloc_device<uint8_t>(sb, q);
        bf16* dx = sycl::malloc_device<bf16>(hx.size(), q);
        const int ncols = epi == 1 ? Ne / 2 : Ne;
        const size_t outb = size_t(R) * ncols * (epi == 1 ? 2 : 4);
        void* dy = sycl::malloc_device<uint8_t>(outb, q);
        q.memcpy(dp, hp.data(), pb); q.memcpy(ds, hs.data(), sb);
        q.memcpy(dx, hx.data(), hx.size() * 2).wait();
        const double flops = 2.0 * R * Ne * K;
        std::vector<uint8_t> first;
        for (int v = 0; v < 3; ++v) {
            // v0: per-thread dequant (the engine kernel), v1/v2: SLM-staged, 32 / 64 rows per thread
            const int scale = 1, rbn = v == 2 ? 8 : 4;
            auto launch = [&]() {
#define MOE_L(EP, SC, RB) moe_es<EP, SC, RB>(q, dp, ds, lut, Ne, K, E * Ne, dx, dy, d_te, d_tmb, d_off, d_cnt, T)
#define MOE_L2(EP, RB) moe_es2<EP, RB>(q, dp, ds, lut, Ne, K, E * Ne, dx, dy, d_te, d_tmb, d_off, d_cnt, T)
                if (epi == 1)
                    return v == 0 ? MOE_L(1, 1, 4) : v == 1 ? MOE_L2(1, 4) : MOE_L2(1, 8);
                return v == 0 ? MOE_L(0, 1, 4) : v == 1 ? MOE_L2(0, 4) : MOE_L2(0, 8);
#undef MOE_L
#undef MOE_L2
            };
            q.memset(dy, 0xFF, outb).wait();                          // NaN if a row is skipped
            launch().wait();
            std::vector<uint8_t> hy(outb);
            q.memcpy(hy.data(), dy, outb).wait();
            double maxerr = 0; int checked = 0, bad = 0;
            for (int e = 0; e < E; ++e) {
                if (!cnt[e]) continue;
                const int rows[3] = {0, cnt[e] / 2, cnt[e] - 1};
                for (int ri = 0; ri < 3; ++ri) {
                    const int r = off[e] + rows[ri];
                    for (int c = (e * 7) % 16; c < ncols; c += 16) {
                        auto dot = [&](int wrow, double& mag) {
                            double s = 0; mag = 0;
                            const size_t gr = size_t(e) * Ne + wrow;
                            for (int k = 0; k < K; ++k) {
                                const uint8_t byte = hp[gr * (K / 2) + k / 2];
                                const int nib = (k & 1) ? (byte >> 4) : (byte & 15);
                                const double w = double(e2m1_f(nib)) * std::ldexp(1.0, int(hs[gr * (K / 32) + k / 32]) - 127);
                                const double xv = double(float(hx[size_t(r) * K + k]));
                                s += w * xv; mag += std::fabs(w * xv);
                            }
                            return s;
                        };
                        double ref, tol, got;
                        if (epi == 0) {
                            double mag; ref = dot(c, mag);
                            float gv; std::memcpy(&gv, hy.data() + (size_t(r) * ncols + c) * 4, 4);
                            got = gv; tol = 2e-5 * mag + 1e-30;
                        } else {
                            double mg, mu; const double gt = dot(c, mg), up = dot(ncols + c, mu);
                            ref = gt / (1.0 + std::exp(-gt)) * up;
                            uint16_t hb; std::memcpy(&hb, hy.data() + (size_t(r) * ncols + c) * 2, 2);
                            uint32_t u = uint32_t(hb) << 16; float gv; std::memcpy(&gv, &u, 4);
                            got = gv;
                            tol = 1e-2 * std::fabs(ref) + 1e-4 * (mg * std::fabs(up) + mu * std::fabs(gt)) + 1e-30;
                        }
                        const double err = std::fabs(got - ref);
                        if (!(err <= tol)) { if (bad < 3) std::printf("   bad e=%d row=%d col=%d ref=%g got=%g\n", e, rows[ri], c, ref, got); ++bad; }
                        maxerr = std::max(maxerr, err / (tol > 0 ? tol : 1));
                        ++checked;
                    }
                }
            }
            bool same = true; (void)same;
            if (first.empty()) first = hy; else same = first == hy;
            double best = 1e30;
            for (int i = 0; i < iters; ++i) {
                const auto a = std::chrono::steady_clock::now();
                launch().wait();
                best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
            }
            const bool ok = bad == 0;
            all_ok = all_ok && ok;
            std::printf("%s N=%4d K=%4d  %s rows/thread %d  %7.3f ms  %6.1f TFLOP/s  checked %d bad %d  %s%s\n",
                cs.what, Ne, K, v == 0 ? "regs" : "slm ", 8 * rbn, best, flops / (best * 1e9), checked, bad,
                ok ? "ok" : "FAIL", v == 0 ? "" : (same ? "  bit-identical to v0" : "  DIFFERS from v0"));
            (void)scale;
        }
        sycl::free(dp, q); sycl::free(ds, q); sycl::free(dx, q); sycl::free(dy, q);
    }
    std::printf("%s\n", all_ok ? "MOE ESIMD PROBE PASS" : "MOE ESIMD PROBE FAIL");
    return all_ok ? 0 : 1;
}
