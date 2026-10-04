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

// smallm_probe.cpp -- small-M (1..16 rows) MXFP4 GEMM on the B70's matrix
// unit, weights read ONCE: correctness against a CPU reference and effective
// weight bandwidth.  The verify pass of MTP / DFlash is exactly this shape
// (4-16 activation rows against every weight of the model), and the batched
// GEMV it uses today runs at ~140 GB/s on Qwen3.8-27B's FFN.
//
//   Y[M][N] = X[M][K] (bf16) * W[N][K]^T (MXFP4: E2M1 nibbles low-first,
//             E8M0 scale per 32-element block)
//
// One ESIMD thread owns 64 output columns x all M rows x one K slice:
//   - W: two 2-D block loads per 32 K (64 rows x 16 bytes), dequantized to
//     UNSCALED bf16 in VNNI order (the scale is a power of two, constant over
//     the 32-K block, so it is applied per column to the DPAS result);
//   - X: 2-D block loads of 8 rows x 16 bf16 (rows >= M read as zero);
//   - xmx::dpas<8,8>: 8 rows x 16 columns x 16 K per instruction.
// K slices (split-K) write partials that a small reduce sums in a fixed order,
// so the result is deterministic.
//
// Build (256-GRF, AOT for the B70), from the repo root:
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 \
//     -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -cl-intel-256-GRF-per-thread" \
//     -O3 -std=c++20 tools/smallm_probe.cpp -o bin/smallm_probe
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

// DQ 0: LUT gather (byte -> two unscaled bf16, 1 KB, L1-resident)
// DQ 1: ALU (build the bf16 bits of each nibble arithmetically)
template <int RBN, int DQ, int PF = 4>
sycl::event smallm(sycl::queue& q, const uint8_t* pay, const uint8_t* scl,
                   const uint32_t* lut, const bf16* X, float* part,
                   int M, int N, int K, int KS, int kc) {
    const int tiles = N / 64;
    const int nthreads = tiles * KS;
    constexpr int TPW = 8;
    const int padded = (nthreads + TPW - 1) / TPW * TPW;
    return q.parallel_for(sycl::nd_range<1>(padded, TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        if constexpr (DQ == 2) {
            // stage the 1 KB byte->bf16-pair table in SLM once per work-group:
            // an SLM gather is a short-latency local access, where the global
            // gather of DQ 0 is a full L1 round trip per 16 bytes of weights
            es::slm_init<1024>();
            const int lid = int(it.get_local_id(0));
            es::slm_block_store<uint32_t, 32>(lid * 128,
                es::block_load<uint32_t, 32>(lut + lid * 32));
            es::barrier();
        }
        const int t = int(it.get_global_id(0));
        if (t >= nthreads) return;
        const int tile = t / KS, ks = t % KS;
        const int n0 = tile * 64;
        const int kb = ks * kc;
        const int ke = kb + kc < K ? kb + kc : K;
        const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
        const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
        es::simd<float, RBN * 4 * 128> acc = 0.0f;
        for (int k = kb; k < ke; k += 128) {
            // E8M0 scales of 4 blocks (128 K) for 64 rows: byte b = block b
            es::simd<uint32_t, 32> s0 = es::load_2d<uint32_t, 1, 32>(
                sclw, K / 32 - 1, N - 1, K / 32 - 1, k / 128, n0);
            es::simd<uint32_t, 32> s1 = es::load_2d<uint32_t, 1, 32>(
                sclw, K / 32 - 1, N - 1, K / 32 - 1, k / 128, n0 + 32);
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int kk = k + 32 * b;
                es::simd<bf16, 128> a0[RBN], a1[RBN];
                #pragma unroll
                for (int rb = 0; rb < RBN; ++rb) {
                    a0[rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, kk, rb * 8);
                    a1[rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, kk + 16, rb * 8);
                }
                // Prefetch the NEXT 32-K block's weight tile into L1 while this
                // one is dequantized and multiplied: without it every block
                // waited out a full memory round trip.  PF blocks ahead.
                if (kk + 32 * PF < ke) {
                    constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
                        es::cache_hint_L1<es::cache_hint::cached>,
                        es::cache_hint_L2<es::cache_hint::cached>};
                    es::prefetch_2d<uint32_t, 4, 32>(payw, K / 2 - 1, N - 1, K / 2 - 1,
                        (kk + 32 * PF) / 8, n0, PFH);
                    es::prefetch_2d<uint32_t, 4, 32>(payw, K / 2 - 1, N - 1, K / 2 - 1,
                        (kk + 32 * PF) / 8, n0 + 32, PFH);
                }
                // 64 rows x 16 bytes (32 K): [32 rows][4 dwords] twice
                es::simd<uint32_t, 128> w0 = es::load_2d<uint32_t, 4, 32>(
                    payw, K / 2 - 1, N - 1, K / 2 - 1, kk / 8, n0);
                es::simd<uint32_t, 128> w1 = es::load_2d<uint32_t, 4, 32>(
                    payw, K / 2 - 1, N - 1, K / 2 - 1, kk / 8, n0 + 32);
                #pragma unroll
                for (int cb = 0; cb < 4; ++cb) {
                    const int lr = (cb & 1) * 16;
                    // column scale: E8M0 byte b of this block, as a float
                    es::simd<uint32_t, 16> sw;
                    if (cb < 2) sw = s0.template select<16, 1>(lr);
                    else        sw = s1.template select<16, 1>(lr);
                    es::simd<uint32_t, 16> e = (sw >> (8 * b)) & 0xFFu;
                    es::simd<uint32_t, 16> sbits = e << 23;
                    sbits.merge(es::simd<uint32_t, 16>(0x00400000u), e == 0u);
                    const es::simd<float, 16> sc = sbits.template bit_cast_view<float>().read();
                    es::simd<uint32_t, 128> vb[2];              // VNNI [8 k-pairs][16 n], two K halves
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        #pragma unroll
                        for (int kp = 0; kp < 8; ++kp) {
                            es::simd<uint32_t, 16> wx;
                            if (cb < 2) wx = w0.template select<16, 4>(4 * lr + 2 * h + (kp >> 2));
                            else        wx = w1.template select<16, 4>(4 * lr + 2 * h + (kp >> 2));
                            const es::simd<uint32_t, 16> byte = (wx >> (8 * (kp & 3))) & 0xFFu;
                            es::simd<uint32_t, 16> pair;
                            if constexpr (DQ == 0) {
                                pair = es::gather<uint32_t, 16>(lut, byte * 4u);
                            } else if constexpr (DQ == 2) {
                                pair = es::slm_gather<uint32_t, 16>(byte * 4u);
                            } else {
                                es::simd<uint32_t, 16> v[2];
                                #pragma unroll
                                for (int s = 0; s < 2; ++s) {
                                    const es::simd<uint32_t, 16> nib = s ? (byte >> 4) : (byte & 0xFu);
                                    const es::simd<uint32_t, 16> m = nib & 7u;
                                    es::simd<uint32_t, 16> mag = 0x3F80u + ((m - 2u) << 6);
                                    mag.merge(es::simd<uint32_t, 16>(0x3F00u), m == 1u);
                                    mag.merge(es::simd<uint32_t, 16>(0u), m == 0u);
                                    v[s] = mag | ((nib & 8u) << 12);
                                }
                                pair = v[0] | (v[1] << 16);
                            }
                            vb[h].template select<16, 1>(kp * 16) = pair;
                        }
                    }
                    #pragma unroll
                    for (int rb = 0; rb < RBN; ++rb) {
                        es::simd<float, 128> tmp = 0.0f;
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(
                            tmp, vb[0].template bit_cast_view<bf16>().read(), a0[rb]);
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(
                            tmp, vb[1].template bit_cast_view<bf16>().read(), a1[rb]);
                        #pragma unroll
                        for (int r = 0; r < 8; ++r)
                            acc.template select<16, 1>((rb * 4 + cb) * 128 + 16 * r) +=
                                tmp.template select<16, 1>(16 * r) * sc;
                    }
                }
            }
        }
        float* out = part + size_t(ks) * M * N;
        #pragma unroll
        for (int rb = 0; rb < RBN; ++rb)
            #pragma unroll
            for (int cb = 0; cb < 4; ++cb)
                es::store_2d<float, 16, 8>(out, N * 4 - 1, M - 1, N * 4 - 1, n0 + cb * 16, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>((rb * 4 + cb) * 128)));
    });
}


// v2: whole cache lines per row.  Each 128-K step loads the 64 rows' full
// 64-byte lines (4 MX blocks) with TRANSPOSED 2-D block loads -- [8 dwords][32
// rows] per load -- so a dword column of 16 rows is contiguous in registers and
// every line is consumed in one pass.  v1 read 16 bytes of each line per block,
// and with dozens of threads per core the half-used lines were evicted before
// the next block came back for them (lm_head stuck at ~40 GB/s).
template <int RBN>
sycl::event smallm_v2(sycl::queue& q, const uint8_t* pay, const uint8_t* scl,
                      const uint32_t* lut, const bf16* X, float* part,
                      int M, int N, int K, int KS, int kc) {
    const int tiles = N / 64;
    const int nthreads = tiles * KS;
    constexpr int TPW = 8;
    const int padded = (nthreads + TPW - 1) / TPW * TPW;
    return q.parallel_for(sycl::nd_range<1>(padded, TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<1024>();
        {
            const int lid = int(it.get_local_id(0));
            es::slm_block_store<uint32_t, 32>(lid * 128, es::block_load<uint32_t, 32>(lut + lid * 32));
            es::barrier();
        }
        const int t = int(it.get_global_id(0));
        if (t >= nthreads) return;
        const int tile = t / KS, ks = t % KS;
        const int n0 = tile * 64;
        const int kb = ks * kc;
        const int ke = kb + kc < K ? kb + kc : K;
        const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
        const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
        constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
        es::simd<float, RBN * 4 * 128> acc = 0.0f;
        for (int k = kb; k < ke; k += 128) {
            if (k + 128 < ke) {
                es::prefetch_2d<uint32_t, 16, 32>(payw, K / 2 - 1, N - 1, K / 2 - 1, (k + 128) / 8, n0, PFH);
                es::prefetch_2d<uint32_t, 16, 32>(payw, K / 2 - 1, N - 1, K / 2 - 1, (k + 128) / 8, n0 + 32, PFH);
            }
            // [8 dwords][32 rows]: tw[i][j] = rows n0+32i.., dwords 8j..8j+7 of this 128-K line
            es::simd<uint32_t, 256> tw00 = es::load_2d<uint32_t, 8, 32, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8, n0);
            es::simd<uint32_t, 256> tw01 = es::load_2d<uint32_t, 8, 32, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8 + 8, n0);
            es::simd<uint32_t, 256> tw10 = es::load_2d<uint32_t, 8, 32, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8, n0 + 32);
            es::simd<uint32_t, 256> tw11 = es::load_2d<uint32_t, 8, 32, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8 + 8, n0 + 32);
            es::simd<uint32_t, 32> s0 = es::load_2d<uint32_t, 1, 32>(sclw, K / 32 - 1, N - 1, K / 32 - 1, k / 128, n0);
            es::simd<uint32_t, 32> s1 = es::load_2d<uint32_t, 1, 32>(sclw, K / 32 - 1, N - 1, K / 32 - 1, k / 128, n0 + 32);
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int kk = k + 32 * b;
                es::simd<bf16, 128> a0[RBN], a1[RBN];
                #pragma unroll
                for (int rb = 0; rb < RBN; ++rb) {
                    a0[rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, kk, rb * 8);
                    a1[rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, kk + 16, rb * 8);
                }
                #pragma unroll
                for (int cb = 0; cb < 4; ++cb) {
                    const int lr = (cb & 1) * 16;
                    es::simd<uint32_t, 16> sw;
                    if (cb < 2) sw = s0.template select<16, 1>(lr);
                    else        sw = s1.template select<16, 1>(lr);
                    es::simd<uint32_t, 16> e = (sw >> (8 * b)) & 0xFFu;
                    es::simd<uint32_t, 16> sbits = e << 23;
                    sbits.merge(es::simd<uint32_t, 16>(0x00400000u), e == 0u);
                    const es::simd<float, 16> sc = sbits.template bit_cast_view<float>().read();
                    es::simd<uint32_t, 128> vb[2];
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        #pragma unroll
                        for (int kp = 0; kp < 8; ++kp) {
                            // dword (within the 16-dword line) of k-pair kp, half h, block b
                            const int dw = 4 * b + 2 * h + (kp >> 2);
                            const int dl = (dw & 7) * 32 + lr;       // [8 dwords][32 rows]
                            es::simd<uint32_t, 16> wx;
                            if (cb < 2) { if (dw < 8) wx = tw00.template select<16, 1>(dl);
                                          else        wx = tw01.template select<16, 1>(dl); }
                            else        { if (dw < 8) wx = tw10.template select<16, 1>(dl);
                                          else        wx = tw11.template select<16, 1>(dl); }
                            const es::simd<uint32_t, 16> byte = (wx >> (8 * (kp & 3))) & 0xFFu;
                            vb[h].template select<16, 1>(kp * 16) = es::slm_gather<uint32_t, 16>(byte * 4u);
                        }
                    }
                    #pragma unroll
                    for (int rb = 0; rb < RBN; ++rb) {
                        es::simd<float, 128> tmp = 0.0f;
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(
                            tmp, vb[0].template bit_cast_view<bf16>().read(), a0[rb]);
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(
                            tmp, vb[1].template bit_cast_view<bf16>().read(), a1[rb]);
                        #pragma unroll
                        for (int r = 0; r < 8; ++r)
                            acc.template select<16, 1>((rb * 4 + cb) * 128 + 16 * r) +=
                                tmp.template select<16, 1>(16 * r) * sc;
                    }
                }
            }
        }
        float* out = part + size_t(ks) * M * N;
        #pragma unroll
        for (int rb = 0; rb < RBN; ++rb)
            #pragma unroll
            for (int cb = 0; cb < 4; ++cb)
                es::store_2d<float, 16, 8>(out, N * 4 - 1, M - 1, N * 4 - 1, n0 + cb * 16, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>((rb * 4 + cb) * 128)));
    });
}


// v3 (2026-10-03): no table at all.  A byte (two E2M1 nibbles, k even low)
// becomes one VNNI dword of two bf16 in the ALU: the nibbles are spread to
// bits 0..3 and 16..19, turned into fp16 bit patterns (magnitude bits at
// 9..11, sign at 15: value * 2^-14, subnormal 0.5 included), widened to fp32
// by the hardware conversion and truncated to bf16 (exact: at most two
// significant bits).  The 2^14 goes into the per-column E8M0 scale, which is
// applied to each 32-K block's DPAS result as before.  One thread owns 32
// columns (two DPAS N-blocks) x all M rows x one K slice; weights come in as
// whole 64-byte lines with transposed 2-D block loads, the next line
// prefetched.  CT = columns per thread (16 or 32).
template <int RBN, int CT>
sycl::event smallm_v3(sycl::queue& q, const uint8_t* pay, const uint8_t* scl,
                      const bf16* X, float* part, int M, int N, int K, int KS, int kc) {
    constexpr int NCB = CT / 16;
    const int tiles = N / CT;
    const int nthreads = tiles * KS;
    constexpr int TPW = 8;
    const int padded = (nthreads + TPW - 1) / TPW * TPW;
    return q.parallel_for(sycl::nd_range<1>(padded, TPW), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        const int t = int(it.get_global_id(0));
        if (t >= nthreads) return;
        const int tile = t % tiles, ks = t / tiles;
        const int n0 = tile * CT;
        const int kb = ks * kc;
        const int ke = kb + kc < K ? kb + kc : K;
        const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
        const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
        constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
        es::simd<float, RBN * NCB * 128> acc = 0.0f;
        for (int k = kb; k < ke; k += 128) {
            if (k + 128 < ke)
                es::prefetch_2d<uint32_t, 16, CT>(payw, K / 2 - 1, N - 1, K / 2 - 1, (k + 128) / 8, n0, PFH);
            // [8 dwords][CT rows] twice: the CT rows' whole 64-byte lines of this 128-K step
            es::simd<uint32_t, 8 * CT> tw0 =
                es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8, n0);
            es::simd<uint32_t, 8 * CT> tw1 =
                es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8 + 8, n0);
            es::simd<uint32_t, CT> sw = es::load_2d<uint32_t, 1, CT>(sclw, K / 32 - 1, N - 1, K / 32 - 1, k / 128, n0);
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int kk = k + 32 * b;
                es::simd<bf16, 128> a0[RBN], a1[RBN];
                #pragma unroll
                for (int rb = 0; rb < RBN; ++rb) {
                    a0[rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, kk, rb * 8);
                    a1[rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, kk + 16, rb * 8);
                }
                #pragma unroll
                for (int cb = 0; cb < NCB; ++cb) {
                    // E8M0 byte b of this block for the 16 columns, times 2^14
                    es::simd<uint32_t, 16> e = (sw.template select<16, 1>(16 * cb) >> (8 * b)) & 0xFFu;
                    // e = 0 (2^-127) becomes the normal 2^-113 here: no special case
                    es::simd<uint32_t, 16> sbits = (e + 14u) << 23;
                    es::simd<float, 16> sc = sbits.template bit_cast_view<float>().read();
                    es::simd<uint32_t, 128> vb[2];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {                        // 4 dwords = 32 K of this block
                        const int dw = 4 * b + j;                          // dword within the 16-dword line
                        es::simd<uint32_t, 16> w;
                        if (dw < 8) w = tw0.template select<16, 1>(dw * CT + 16 * cb);
                        else        w = tw1.template select<16, 1>((dw - 8) * CT + 16 * cb);
                        #pragma unroll
                        for (int qb = 0; qb < 4; ++qb) {                   // byte qb = k-pair 4j+qb
                            es::simd<uint32_t, 16> tb = qb ? (w >> (8 * qb)) : w;
                            es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                            es::simd<uint32_t, 16> h = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                            es::simd<sycl::half, 32> hv = h.template bit_cast_view<sycl::half>().read();
                            es::simd<float, 32> fv = hv;
                            es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                            es::simd<uint16_t, 32> bb = fb >> 16;
                            const int kp = 4 * j + qb;                     // 0..15 within the block
                            vb[kp >> 3].template select<16, 1>((kp & 7) * 16) =
                                bb.template bit_cast_view<uint32_t>().read();
                        }
                    }
                    #pragma unroll
                    for (int rb = 0; rb < RBN; ++rb) {
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
        float* out = part + size_t(ks) * M * N;
        #pragma unroll
        for (int rb = 0; rb < RBN; ++rb)
            #pragma unroll
            for (int cb = 0; cb < NCB; ++cb)
                es::store_2d<float, 16, 8>(out, N * 4 - 1, M - 1, N * 4 - 1, n0 + cb * 16, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>((rb * NCB + cb) * 128)));
    });
}


// v4: v3's ALU decode + DPAS, 16 columns per thread, with split-K kept INSIDE a
// work-group: the KS threads of one 16-column tile each run a K slice, put
// their M x 16 partial in SLM, and thread 0 sums them in a fixed order and
// writes Y.  v3 wrote KS x M x N partials to global memory and needed a
// second kernel: at M = 8 that traffic rivalled the weights on the 2-8K-wide
// projections and grew with M.  Deterministic (fixed summation order).
template <int RBN>
sycl::event smallm_v4(sycl::queue& q, const uint8_t* pay, const uint8_t* scl,
                      const bf16* X, float* Y, int M, int N, int K, int KS, int kc) {
    const int tiles = N / 16;
    return q.parallel_for(sycl::nd_range<1>(size_t(tiles) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<16 * RBN * 128 * 4>();
        const int ks = int(it.get_local_id(0));
        const int tile = int(it.get_group(0));
        const int n0 = tile * 16;
        const int kb = ks * kc;
        const int ke = kb + kc < K ? kb + kc : K;
        const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
        const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
        constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
        es::simd<float, RBN * 128> acc = 0.0f;
        for (int k = kb; k < ke; k += 128) {
            if (k + 128 < ke)
                es::prefetch_2d<uint32_t, 16, 16>(payw, K / 2 - 1, N - 1, K / 2 - 1, (k + 128) / 8, n0, PFH);
            es::simd<uint32_t, 128> tw0 =
                es::load_2d<uint32_t, 8, 16, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8, n0);
            es::simd<uint32_t, 128> tw1 =
                es::load_2d<uint32_t, 8, 16, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8 + 8, n0);
            es::simd<uint32_t, 16> sw = es::load_2d<uint32_t, 1, 16>(sclw, K / 32 - 1, N - 1, K / 32 - 1, k / 128, n0);
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int kk = k + 32 * b;
                es::simd<bf16, 128> a0[RBN], a1[RBN];
                #pragma unroll
                for (int rb = 0; rb < RBN; ++rb) {
                    a0[rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, kk, rb * 8);
                    a1[rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, kk + 16, rb * 8);
                }
                es::simd<uint32_t, 16> e = (sw >> (8 * b)) & 0xFFu;
                es::simd<uint32_t, 16> sbits = (e + 14u) << 23;
                es::simd<float, 16> sc = sbits.template bit_cast_view<float>().read();
                es::simd<uint32_t, 128> vb[2];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int dw = 4 * b + j;
                    es::simd<uint32_t, 16> w;
                    if (dw < 8) w = tw0.template select<16, 1>(dw * 16);
                    else        w = tw1.template select<16, 1>((dw - 8) * 16);
                    #pragma unroll
                    for (int qb = 0; qb < 4; ++qb) {
                        es::simd<uint32_t, 16> tb = qb ? (w >> (8 * qb)) : w;
                        es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                        es::simd<uint32_t, 16> h = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                        es::simd<sycl::half, 32> hv = h.template bit_cast_view<sycl::half>().read();
                        es::simd<float, 32> fv = hv;
                        es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                        es::simd<uint16_t, 32> bb = fb >> 16;
                        const int kp = 4 * j + qb;
                        vb[kp >> 3].template select<16, 1>((kp & 7) * 16) = bb.template bit_cast_view<uint32_t>().read();
                    }
                }
                #pragma unroll
                for (int rb = 0; rb < RBN; ++rb) {
                    es::simd<float, 128> tmp = 0.0f;
                    tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, vb[0].template bit_cast_view<bf16>().read(), a0[rb]);
                    tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, vb[1].template bit_cast_view<bf16>().read(), a1[rb]);
                    #pragma unroll
                    for (int r = 0; r < 8; ++r)
                        acc.template select<16, 1>(rb * 128 + 16 * r) += tmp.template select<16, 1>(16 * r) * sc;
                }
            }
        }
        if (KS > 1) {
            es::slm_block_store<float, RBN * 128>(ks * RBN * 128 * 4, acc);
            es::barrier();
            if (ks != 0) return;
            for (int i = 1; i < KS; ++i) acc += es::slm_block_load<float, RBN * 128>(i * RBN * 128 * 4);
        }
        #pragma unroll
        for (int rb = 0; rb < RBN; ++rb)
            es::store_2d<float, 16, 8>(Y, N * 4 - 1, M - 1, N * 4 - 1, n0, rb * 8,
                es::simd<float, 128>(acc.template select<128, 1>(rb * 128)));
    });
}
struct Plan4 { int KS, kc; };
static int g_target4 = 4096;
static Plan4 plan4(int N, int K) {
    const int tiles = N / 16;
    int KS = 1;
    while (KS < 16 && tiles * KS < g_target4 && K / (KS * 2) >= 128) KS *= 2;
    int kc = (K + KS - 1) / KS; kc = (kc + 127) / 128 * 128;
    KS = (K + kc - 1) / kc;
    return {KS, kc};
}


// v5: v4 with CT columns per thread (16/32/64: one A load feeds CT/16 DPAS
// column blocks, cutting the activation re-reads that made v4 slow down with
// M on 2-8K-wide matrices) and TPT tiles per work-group (so a KS = 1 shape
// still gets 8-thread work-groups).  Work-group = TPT tiles x KS slices;
// split-K partials reduced in SLM in a fixed order.
template <int RBN, int CT>
sycl::event smallm_v5(sycl::queue& q, const uint8_t* pay, const uint8_t* scl,
                      const bf16* X, float* Y, int M, int N, int K, int KS, int kc, int TPT) {
    constexpr int NCB = CT / 16;
    const int tiles = N / CT;
    const int groups = (tiles + TPT - 1) / TPT;
    return q.parallel_for(sycl::nd_range<1>(size_t(groups) * TPT * KS, size_t(TPT) * KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<32 * RBN * NCB * 128 * 4>();
        const int lid = int(it.get_local_id(0));
        const int tw = lid / KS, ks = lid % KS;
        const int tile = int(it.get_group(0)) * TPT + tw;
        const bool live = tile < tiles;
        const int n0 = (live ? tile : 0) * CT;
        const int kb = ks * kc;
        const int ke = kb + kc < K ? kb + kc : K;
        const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
        const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
        constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
        es::simd<float, RBN * NCB * 128> acc = 0.0f;
        if (live) {
        for (int k = kb; k < ke; k += 128) {
            if (k + 128 < ke)
                es::prefetch_2d<uint32_t, 16, CT>(payw, K / 2 - 1, N - 1, K / 2 - 1, (k + 128) / 8, n0, PFH);
            es::simd<uint32_t, 8 * CT> tw0 =
                es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8, n0);
            es::simd<uint32_t, 8 * CT> tw1 =
                es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8 + 8, n0);
            es::simd<uint32_t, CT> sw = es::load_2d<uint32_t, 1, CT>(sclw, K / 32 - 1, N - 1, K / 32 - 1, k / 128, n0);
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int kk = k + 32 * b;
                es::simd<bf16, 128> a0[RBN], a1[RBN];
                #pragma unroll
                for (int rb = 0; rb < RBN; ++rb) {
                    a0[rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, kk, rb * 8);
                    a1[rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, kk + 16, rb * 8);
                }
                #pragma unroll
                for (int cb = 0; cb < NCB; ++cb) {
                    es::simd<uint32_t, 16> e = (sw.template select<16, 1>(16 * cb) >> (8 * b)) & 0xFFu;
                    es::simd<uint32_t, 16> sbits = (e + 14u) << 23;
                    es::simd<float, 16> sc = sbits.template bit_cast_view<float>().read();
                    es::simd<uint32_t, 128> vb[2];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int dw = 4 * b + j;
                        es::simd<uint32_t, 16> w;
                        if (dw < 8) w = tw0.template select<16, 1>(dw * CT + 16 * cb);
                        else        w = tw1.template select<16, 1>((dw - 8) * CT + 16 * cb);
                        #pragma unroll
                        for (int qb = 0; qb < 4; ++qb) {
                            es::simd<uint32_t, 16> tb = qb ? (w >> (8 * qb)) : w;
                            es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                            es::simd<uint32_t, 16> h = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                            es::simd<sycl::half, 32> hv = h.template bit_cast_view<sycl::half>().read();
                            es::simd<float, 32> fv = hv;
                            es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                            es::simd<uint16_t, 32> bb = fb >> 16;
                            const int kp = 4 * j + qb;
                            vb[kp >> 3].template select<16, 1>((kp & 7) * 16) = bb.template bit_cast_view<uint32_t>().read();
                        }
                    }
                    #pragma unroll
                    for (int rb = 0; rb < RBN; ++rb) {
                        es::simd<float, 128> tmp = 0.0f;
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, vb[0].template bit_cast_view<bf16>().read(), a0[rb]);
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, vb[1].template bit_cast_view<bf16>().read(), a1[rb]);
                        #pragma unroll
                        for (int r = 0; r < 8; ++r)
                            acc.template select<16, 1>((rb * NCB + cb) * 128 + 16 * r) += tmp.template select<16, 1>(16 * r) * sc;
                    }
                }
            }
        }
        }
        if (KS > 1) {
            es::slm_block_store<float, RBN * NCB * 128>(lid * RBN * NCB * 128 * 4, acc);
            es::barrier();
            if (ks != 0 || !live) return;
            for (int i = 1; i < KS; ++i)
                acc += es::slm_block_load<float, RBN * NCB * 128>((tw * KS + i) * RBN * NCB * 128 * 4);
        } else if (!live) return;
        #pragma unroll
        for (int rb = 0; rb < RBN; ++rb)
            #pragma unroll
            for (int cb = 0; cb < NCB; ++cb)
                es::store_2d<float, 16, 8>(Y, N * 4 - 1, M - 1, N * 4 - 1, n0 + 16 * cb, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>((rb * NCB + cb) * 128)));
    });
}
template <int RBN, int CT>
sycl::event smallm_v6(sycl::queue& q, const uint8_t* pay, const uint8_t* scl,
                      const bf16* X, float* Y, int M, int N, int K, int KS, int kc, int TPT) {
    constexpr int NCB = CT / 16;
    const int tiles = N / CT;
    const int groups = (tiles + TPT - 1) / TPT;
    return q.parallel_for(sycl::nd_range<1>(size_t(groups) * TPT * KS, size_t(TPT) * KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<32 * RBN * NCB * 128 * 4>();
        const int lid = int(it.get_local_id(0));
        const int tw = lid / KS, ks = lid % KS;
        const int tile = int(it.get_group(0)) * TPT + tw;
        const bool live = tile < tiles;
        const int n0 = (live ? tile : 0) * CT;
        const int kb = ks * kc;
        const int ke = kb + kc < K ? kb + kc : K;
        const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
        const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
        constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
        es::simd<float, RBN * NCB * 128> acc = 0.0f;
        if (live) {
        for (int k = kb; k < ke; k += 128) {
            if (k + 128 < ke)
                es::prefetch_2d<uint32_t, 16, CT>(payw, K / 2 - 1, N - 1, K / 2 - 1, (k + 128) / 8, n0, PFH);
            es::simd<uint32_t, 8 * CT> tw0 =
                es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8, n0);
            es::simd<uint32_t, 8 * CT> tw1 =
                es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8 + 8, n0);
            es::simd<uint32_t, CT> sw = es::load_2d<uint32_t, 1, CT>(sclw, K / 32 - 1, N - 1, K / 32 - 1, k / 128, n0);
            // every activation tile of this 128-K step in flight together with
            // the weight lines, instead of one load-to-use wait per 32-K block
            es::simd<bf16, 128> aa[4][2][RBN];
            #pragma unroll
            for (int b = 0; b < 4; ++b)
                #pragma unroll
                for (int rb = 0; rb < RBN; ++rb) {
                    aa[b][0][rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, k + 32 * b, rb * 8);
                    aa[b][1][rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, k + 32 * b + 16, rb * 8);
                }
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                #pragma unroll
                for (int cb = 0; cb < NCB; ++cb) {
                    es::simd<uint32_t, 16> e = (sw.template select<16, 1>(16 * cb) >> (8 * b)) & 0xFFu;
                    es::simd<uint32_t, 16> sbits = (e + 14u) << 23;
                    es::simd<float, 16> sc = sbits.template bit_cast_view<float>().read();
                    es::simd<uint32_t, 128> vb[2];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int dw = 4 * b + j;
                        es::simd<uint32_t, 16> w;
                        if (dw < 8) w = tw0.template select<16, 1>(dw * CT + 16 * cb);
                        else        w = tw1.template select<16, 1>((dw - 8) * CT + 16 * cb);
                        #pragma unroll
                        for (int qb = 0; qb < 4; ++qb) {
                            es::simd<uint32_t, 16> tb = qb ? (w >> (8 * qb)) : w;
                            es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                            es::simd<uint32_t, 16> h = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                            es::simd<sycl::half, 32> hv = h.template bit_cast_view<sycl::half>().read();
                            es::simd<float, 32> fv = hv;
                            es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                            es::simd<uint16_t, 32> bb = fb >> 16;
                            const int kp = 4 * j + qb;
                            vb[kp >> 3].template select<16, 1>((kp & 7) * 16) = bb.template bit_cast_view<uint32_t>().read();
                        }
                    }
                    #pragma unroll
                    for (int rb = 0; rb < RBN; ++rb) {
                        es::simd<float, 128> tmp = 0.0f;
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, vb[0].template bit_cast_view<bf16>().read(), aa[b][0][rb]);
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, vb[1].template bit_cast_view<bf16>().read(), aa[b][1][rb]);
                        #pragma unroll
                        for (int r = 0; r < 8; ++r)
                            acc.template select<16, 1>((rb * NCB + cb) * 128 + 16 * r) += tmp.template select<16, 1>(16 * r) * sc;
                    }
                }
            }
        }
        }
        if (KS > 1) {
            es::slm_block_store<float, RBN * NCB * 128>(lid * RBN * NCB * 128 * 4, acc);
            es::barrier();
            if (ks != 0 || !live) return;
            for (int i = 1; i < KS; ++i)
                acc += es::slm_block_load<float, RBN * NCB * 128>((tw * KS + i) * RBN * NCB * 128 * 4);
        } else if (!live) return;
        #pragma unroll
        for (int rb = 0; rb < RBN; ++rb)
            #pragma unroll
            for (int cb = 0; cb < NCB; ++cb)
                es::store_2d<float, 16, 8>(Y, N * 4 - 1, M - 1, N * 4 - 1, n0 + 16 * cb, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>((rb * NCB + cb) * 128)));
    });
}
template <int RBN, int CT>
sycl::event smallm_v7(sycl::queue& q, const uint8_t* pay, const uint8_t* scl,
                      const bf16* X, float* Y, int M, int N, int K, int KS, int kc, int TPT) {
    constexpr int NCB = CT / 16;
    const int tiles = N / CT;
    const int groups = (tiles + TPT - 1) / TPT;
    return q.parallel_for(sycl::nd_range<1>(size_t(groups) * TPT * KS, size_t(TPT) * KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<32 * RBN * NCB * 128 * 4>();
        const int lid = int(it.get_local_id(0));
        const int tw = lid / KS, ks = lid % KS;
        const int tile = int(it.get_group(0)) * TPT + tw;
        const bool live = tile < tiles;
        const int n0 = (live ? tile : 0) * CT;
        const int kb = ks * kc;
        const int ke = kb + kc < K ? kb + kc : K;
        const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
        const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
        constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
        es::simd<float, RBN * NCB * 128> acc = 0.0f;
        if (live) {
        for (int k = kb; k < ke; k += 128) {
            if (k + 128 < ke)
                es::prefetch_2d<uint32_t, 16, CT>(payw, K / 2 - 1, N - 1, K / 2 - 1, (k + 128) / 8, n0, PFH);
            es::simd<uint32_t, 8 * CT> tw0 =
                es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8, n0);
            es::simd<uint32_t, 8 * CT> tw1 =
                es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8 + 8, n0);
            es::simd<uint32_t, CT> sw = es::load_2d<uint32_t, 1, CT>(sclw, K / 32 - 1, N - 1, K / 32 - 1, k / 128, n0);
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int kk = k + 32 * b;
                // one array load: [2 blocks][8 rows][16 bf16] = both 16-K halves,
                // whole 64-byte lines, one message instead of two
                es::simd<bf16, 256> a01[RBN];
                #pragma unroll
                for (int rb = 0; rb < RBN; ++rb)
                    a01[rb] = es::load_2d<bf16, 16, 8, 2>(X, K * 2 - 1, M - 1, K * 2 - 1, kk, rb * 8);
                #pragma unroll
                for (int cb = 0; cb < NCB; ++cb) {
                    es::simd<uint32_t, 16> e = (sw.template select<16, 1>(16 * cb) >> (8 * b)) & 0xFFu;
                    es::simd<uint32_t, 16> sbits = (e + 14u) << 23;
                    es::simd<float, 16> sc = sbits.template bit_cast_view<float>().read();
                    es::simd<uint32_t, 128> vb[2];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int dw = 4 * b + j;
                        es::simd<uint32_t, 16> w;
                        if (dw < 8) w = tw0.template select<16, 1>(dw * CT + 16 * cb);
                        else        w = tw1.template select<16, 1>((dw - 8) * CT + 16 * cb);
                        #pragma unroll
                        for (int qb = 0; qb < 4; ++qb) {
                            es::simd<uint32_t, 16> tb = qb ? (w >> (8 * qb)) : w;
                            es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                            es::simd<uint32_t, 16> h = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                            es::simd<sycl::half, 32> hv = h.template bit_cast_view<sycl::half>().read();
                            es::simd<float, 32> fv = hv;
                            es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                            es::simd<uint16_t, 32> bb = fb >> 16;
                            const int kp = 4 * j + qb;
                            vb[kp >> 3].template select<16, 1>((kp & 7) * 16) = bb.template bit_cast_view<uint32_t>().read();
                        }
                    }
                    #pragma unroll
                    for (int rb = 0; rb < RBN; ++rb) {
                        es::simd<float, 128> tmp = 0.0f;
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, vb[0].template bit_cast_view<bf16>().read(), es::simd<bf16, 128>(a01[rb].template select<128, 1>(0)));
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, vb[1].template bit_cast_view<bf16>().read(), es::simd<bf16, 128>(a01[rb].template select<128, 1>(128)));
                        #pragma unroll
                        for (int r = 0; r < 8; ++r)
                            acc.template select<16, 1>((rb * NCB + cb) * 128 + 16 * r) += tmp.template select<16, 1>(16 * r) * sc;
                    }
                }
            }
        }
        }
        if (KS > 1) {
            es::slm_block_store<float, RBN * NCB * 128>(lid * RBN * NCB * 128 * 4, acc);
            es::barrier();
            if (ks != 0 || !live) return;
            for (int i = 1; i < KS; ++i)
                acc += es::slm_block_load<float, RBN * NCB * 128>((tw * KS + i) * RBN * NCB * 128 * 4);
        } else if (!live) return;
        #pragma unroll
        for (int rb = 0; rb < RBN; ++rb)
            #pragma unroll
            for (int cb = 0; cb < NCB; ++cb)
                es::store_2d<float, 16, 8>(Y, N * 4 - 1, M - 1, N * 4 - 1, n0 + 16 * cb, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>((rb * NCB + cb) * 128)));
    });
}
template <int RBN, int CT>
sycl::event smallm_v8(sycl::queue& q, const uint8_t* pay, const uint8_t* scl,
                      const bf16* X, float* Y, int M, int N, int K, int KS, int kc, int TPT) {
    constexpr int NCB = CT / 16;
    const int tiles = N / CT;
    const int groups = (tiles + TPT - 1) / TPT;
    return q.parallel_for(sycl::nd_range<1>(size_t(groups) * TPT * KS, size_t(TPT) * KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<32 * RBN * NCB * 128 * 4>();
        const int lid = int(it.get_local_id(0));
        const int tw = lid / KS, ks = lid % KS;
        const int tile = int(it.get_group(0)) * TPT + tw;
        const bool live = tile < tiles;
        const int n0 = (live ? tile : 0) * CT;
        const int kb = ks * kc;
        const int ke = kb + kc < K ? kb + kc : K;
        const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
        const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
        constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
        constexpr auto AH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
        constexpr auto WH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::uncached>, es::cache_hint_L2<es::cache_hint::cached>};
        es::simd<float, RBN * NCB * 128> acc = 0.0f;
        if (live) {
        for (int k = kb; k < ke; k += 128) {
            if (k + 128 < ke)
                es::prefetch_2d<uint32_t, 16, CT>(payw, K / 2 - 1, N - 1, K / 2 - 1, (k + 128) / 8, n0, WH);
            es::simd<uint32_t, 8 * CT> tw0 =
                es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8, n0, WH);
            es::simd<uint32_t, 8 * CT> tw1 =
                es::load_2d<uint32_t, 8, CT, 1, true, false>(payw, K / 2 - 1, N - 1, K / 2 - 1, k / 8 + 8, n0, WH);
            es::simd<uint32_t, CT> sw = es::load_2d<uint32_t, 1, CT>(sclw, K / 32 - 1, N - 1, K / 32 - 1, k / 128, n0);
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int kk = k + 32 * b;
                es::simd<bf16, 128> a0[RBN], a1[RBN];
                #pragma unroll
                for (int rb = 0; rb < RBN; ++rb) {
                    a0[rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, kk, rb * 8, AH);
                    a1[rb] = es::load_2d<bf16, 16, 8>(X, K * 2 - 1, M - 1, K * 2 - 1, kk + 16, rb * 8, AH);
                }
                #pragma unroll
                for (int cb = 0; cb < NCB; ++cb) {
                    es::simd<uint32_t, 16> e = (sw.template select<16, 1>(16 * cb) >> (8 * b)) & 0xFFu;
                    es::simd<uint32_t, 16> sbits = (e + 14u) << 23;
                    es::simd<float, 16> sc = sbits.template bit_cast_view<float>().read();
                    es::simd<uint32_t, 128> vb[2];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int dw = 4 * b + j;
                        es::simd<uint32_t, 16> w;
                        if (dw < 8) w = tw0.template select<16, 1>(dw * CT + 16 * cb);
                        else        w = tw1.template select<16, 1>((dw - 8) * CT + 16 * cb);
                        #pragma unroll
                        for (int qb = 0; qb < 4; ++qb) {
                            es::simd<uint32_t, 16> tb = qb ? (w >> (8 * qb)) : w;
                            es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                            es::simd<uint32_t, 16> h = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                            es::simd<sycl::half, 32> hv = h.template bit_cast_view<sycl::half>().read();
                            es::simd<float, 32> fv = hv;
                            es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                            es::simd<uint16_t, 32> bb = fb >> 16;
                            const int kp = 4 * j + qb;
                            vb[kp >> 3].template select<16, 1>((kp & 7) * 16) = bb.template bit_cast_view<uint32_t>().read();
                        }
                    }
                    #pragma unroll
                    for (int rb = 0; rb < RBN; ++rb) {
                        es::simd<float, 128> tmp = 0.0f;
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, vb[0].template bit_cast_view<bf16>().read(), a0[rb]);
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, vb[1].template bit_cast_view<bf16>().read(), a1[rb]);
                        #pragma unroll
                        for (int r = 0; r < 8; ++r)
                            acc.template select<16, 1>((rb * NCB + cb) * 128 + 16 * r) += tmp.template select<16, 1>(16 * r) * sc;
                    }
                }
            }
        }
        }
        if (KS > 1) {
            es::slm_block_store<float, RBN * NCB * 128>(lid * RBN * NCB * 128 * 4, acc);
            es::barrier();
            if (ks != 0 || !live) return;
            for (int i = 1; i < KS; ++i)
                acc += es::slm_block_load<float, RBN * NCB * 128>((tw * KS + i) * RBN * NCB * 128 * 4);
        } else if (!live) return;
        #pragma unroll
        for (int rb = 0; rb < RBN; ++rb)
            #pragma unroll
            for (int cb = 0; cb < NCB; ++cb)
                es::store_2d<float, 16, 8>(Y, N * 4 - 1, M - 1, N * 4 - 1, n0 + 16 * cb, rb * 8,
                    es::simd<float, 128>(acc.template select<128, 1>((rb * NCB + cb) * 128)));
    });
}
// "AW": weights as the DPAS A operand (8 output rows x 16 K per DPAS), read as
// 64-byte-wide 2-D tiles (8 rows x 128 K); X as the B operand via transposed
// 2-D loads (VNNI [8 k-pairs][16 tokens], tokens >= M read as zero).  RBW
// row-blocks of 8 outputs per thread.  The grouped MoE gate_up went 297 -> 380
// GB/s with this layout (moe_smallm_probe).
template <int RBW>
sycl::event smallm_aw(sycl::queue& q, const uint8_t* pay, const uint8_t* scl,
                      const bf16* X, float* Y, int M, int N, int K, int KS, int kc, int TPT) {
    constexpr int ACC = RBW * 128;
    const int tiles = N / (8 * RBW);
    const int groups = (tiles + TPT - 1) / TPT;
    const uint32_t* payw = reinterpret_cast<const uint32_t*>(pay);
    const uint32_t* sclw = reinterpret_cast<const uint32_t*>(scl);
    const uint32_t* Xw = reinterpret_cast<const uint32_t*>(X);
    const unsigned PW = unsigned(K) / 2 - 1, PH = unsigned(N) - 1, PP = unsigned(K) / 2 - 1;
    const unsigned SW = unsigned(K) / 32 - 1, SP = unsigned(K) / 32 - 1;
    const unsigned XW = unsigned(K) * 2 - 1, XH = unsigned(M) - 1;
    return q.parallel_for(sycl::nd_range<1>(size_t(groups) * TPT * KS, size_t(TPT) * KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<32 * ACC * 4>();
        const int lid = int(it.get_local_id(0));
        const int tw = lid / KS, ks = lid % KS;
        const int tile = int(it.get_group(0)) * TPT + tw;
        const bool live = tile < tiles;
        const int c0 = (live ? tile : 0) * 8 * RBW;
        const int kb = ks * kc;
        const int ke = kb + kc < K ? kb + kc : K;
        constexpr auto PFH = sycl::ext::oneapi::experimental::properties{
            es::cache_hint_L1<es::cache_hint::cached>, es::cache_hint_L2<es::cache_hint::cached>};
        const es::simd<uint32_t, 16> shv = (es::simd<uint32_t, 16>(0, 1) & 3u) << 3;
        es::simd<float, ACC> acc = 0.0f;
        if (live) {
            for (int k = kb; k < ke; k += 128) {
                es::simd<uint32_t, 128> wt[RBW];
                es::simd<uint32_t, 8> sw[RBW];
                #pragma unroll
                for (int b = 0; b < RBW; ++b) {
                    if (k + 128 < ke)
                        es::prefetch_2d<uint32_t, 16, 8>(payw, PW, PH, PP, (k + 128) / 8, c0 + 8 * b, PFH);
                    wt[b] = es::load_2d<uint32_t, 16, 8>(payw, PW, PH, PP, k / 8, c0 + 8 * b);
                    sw[b] = es::load_2d<uint32_t, 1, 8>(sclw, SW, PH, SP, k / 128, c0 + 8 * b);
                }
                #pragma unroll
                for (int g = 0; g < 4; ++g) {
                    es::simd<uint32_t, 128> bt0 = es::load_2d<uint32_t, 8, 16, 1, true, false>(Xw, XW, XH, XW, (k + 32 * g) / 2, 0);
                    es::simd<uint32_t, 128> bt1 = es::load_2d<uint32_t, 8, 16, 1, true, false>(Xw, XW, XH, XW, (k + 32 * g) / 2 + 8, 0);
                    #pragma unroll
                    for (int b = 0; b < RBW; ++b) {
                        es::simd<uint32_t, 64> a0, a1;
                        #pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            es::simd<uint32_t, 16> rep = wt[b].template replicate_vs_w_hs<4, 1, 4, 0>(r * 16 + 4 * g);
                            es::simd<uint32_t, 16> tb = rep >> shv;
                            es::simd<uint32_t, 16> u = (tb & 0x0Fu) | ((tb & 0xF0u) << 12);
                            es::simd<uint32_t, 16> hb = ((u & 0x00070007u) << 9) | ((u & 0x00080008u) << 12);
                            es::simd<sycl::half, 32> hv = hb.template bit_cast_view<sycl::half>().read();
                            es::simd<float, 32> fv = hv;
                            es::simd<uint32_t, 32> fb = fv.template bit_cast_view<uint32_t>().read();
                            es::simd<uint16_t, 32> bb = fb >> 16;
                            es::simd<uint32_t, 16> pr = bb.template bit_cast_view<uint32_t>().read();
                            a0.template select<8, 1>(r * 8) = pr.template select<8, 1>(0);
                            a1.template select<8, 1>(r * 8) = pr.template select<8, 1>(8);
                        }
                        es::simd<float, 128> tmp = 0.0f;
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, bt0.template bit_cast_view<bf16>().read(), a0.template bit_cast_view<bf16>().read());
                        tmp = xmx::dpas<8, 8, float, float, bf16, bf16>(tmp, bt1.template bit_cast_view<bf16>().read(), a1.template bit_cast_view<bf16>().read());
                        es::simd<uint32_t, 8> ev = (sw[b] >> (8 * g)) & 0xFFu;
                        es::simd<uint32_t, 8> sbits = (ev + 14u) << 23;
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
        const int nr = M < 16 ? M : 16;
        for (int n = 0; n < nr; ++n) {
            #pragma unroll
            for (int b = 0; b < RBW; ++b) {
                es::simd<float, 8> col = acc.template select<8, 16>(b * 128 + n);
                es::block_store<float, 8>(Y + size_t(n) * N + c0 + 8 * b, col);
            }
        }
    });
}
struct PlanAW { int KS, kc, TPT; };
static PlanAW plan_aw(int N, int K, int RBW) {
    const int tiles = N / (8 * RBW);
    int KS = 1;
    while (KS < 16 && tiles * KS < 4096 && K / (KS * 2) >= 128) KS *= 2;
    int kc = (K + KS - 1) / KS; kc = (kc + 127) / 128 * 128;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= 32) TPT *= 2;
    return {KS, kc, TPT};
}
struct Plan5 { int KS, kc, TPT; };
static int g_target5 = 4096;
static Plan5 plan5(int N, int K, int CT) {
    const int tiles = N / CT;
    int KS = 1;
    while (KS < 16 && tiles * KS < g_target5 && K / (KS * 2) >= 128) KS *= 2;
    int kc = (K + KS - 1) / KS; kc = (kc + 127) / 128 * 128;
    KS = (K + kc - 1) / kc;
    int TPT = 1;
    while (TPT * KS < 8 && TPT * 2 * KS <= 32) TPT *= 2;
    return {KS, kc, TPT};
}

static sycl::event reduce(sycl::queue& q, const float* part, float* y, int M, int N, int KS,
                          sycl::event dep) {
    return q.parallel_for(sycl::range<1>(size_t(M) * N), dep, [=](sycl::id<1> i) {
        float s = 0.0f;
        for (int ks = 0; ks < KS; ++ks) s += part[size_t(ks) * M * N + i[0]];
        y[i[0]] = s;
    });
}

struct Plan { int KS, kc; };
static int g_target = 2048;
static Plan plan_for(int N, int K) {
    const int tiles = N / 64;
    int KS = std::max(1, (g_target + tiles - 1) / tiles);  // ~g_target threads
    KS = std::min(KS, K / 128);
    int kc = (K + KS - 1) / KS; kc = (kc + 127) / 128 * 128;
    KS = (K + kc - 1) / kc;
    return {KS, kc};
}

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
    if (argc > 2) g_target = std::atoi(argv[2]);
    std::printf("split-K thread target %d\n", g_target);
    // byte -> (bf16(e2m1(hi)) << 16) | bf16(e2m1(lo)): VNNI keeps k even low
    std::vector<uint32_t> hlut(256);
    for (int b = 0; b < 256; ++b)
        hlut[b] = bf16_bits_exact(e2m1_f(b & 15)) | (bf16_bits_exact(e2m1_f(b >> 4)) << 16);
    uint32_t* lut = sycl::malloc_device<uint32_t>(256, q);
    q.memcpy(lut, hlut.data(), 1024).wait();

    struct Shape { int N, K; const char* what; };
    const Shape shapes[] = {
        {8192, 2048, "orn la_qkv"}, {4096, 2048, "orn z / q"}, {2048, 4096, "orn out / o"},
        {1024, 2048, "orn expert gu"}, {2048, 512, "orn expert dn"}, {248320, 2048, "orn lm_head"},
        {5120, 5120, "q27 o_proj"}, {34816, 5120, "q27 ffn gu"}};
    bool all_ok = true;
    std::mt19937 rng(11);
    for (const auto& sh : shapes) {
        const int N = sh.N, K = sh.K;
        const size_t pb = size_t(N) * K / 2, sb = size_t(N) * K / 32;
        std::vector<uint8_t> hp(pb), hs(sb);
        for (auto& x : hp) x = uint8_t(rng());
        for (auto& x : hs) x = uint8_t(118 + rng() % 12);          // 2^-9 .. 2^2
        // rotating copies: >= 512 MB of weights between reuses, so the timed
        // loop streams from DRAM instead of re-reading an L2-resident matrix
        const int NC = int(std::min<size_t>(32, std::max<size_t>(1, (size_t(512) << 20) / (pb + sb))));
        std::vector<uint8_t*> cps(NC), css(NC);
        for (int c = 0; c < NC; ++c) {
            cps[c] = sycl::malloc_device<uint8_t>(pb, q); css[c] = sycl::malloc_device<uint8_t>(sb, q);
            q.memcpy(cps[c], hp.data(), pb); q.memcpy(css[c], hs.data(), sb);
        }
        q.wait();
        uint8_t* dp = cps[0]; uint8_t* ds = css[0];
        for (int M : {1, 2, 4, 8}) {
            std::vector<bf16> hx(size_t(M) * K);
            std::uniform_real_distribution<float> U(-1.f, 1.f);
            for (auto& v : hx) v = bf16(U(rng));
            bf16* dx = sycl::malloc_device<bf16>(hx.size(), q);
            q.memcpy(dx, hx.data(), hx.size() * 2).wait();
            const Plan pl = plan_for(N, K);
            float* part = sycl::malloc_device<float>(size_t(pl.KS) * M * N, q);
            float* dy = sycl::malloc_device<float>(size_t(M) * N, q);
            for (int dq : {7, 13, 14}) {
                int ci = 0;
                auto launch = [&]() {
                    sycl::event e;
                    float* o = pl.KS > 1 ? part : dy;
                    const uint8_t* wp = cps[ci % NC]; const uint8_t* wsc = css[ci % NC]; ++ci;
                    if (dq == 13) { const PlanAW pa = plan_aw(N, K, 1);
                                    return smallm_aw<1>(q, wp, wsc, dx, dy, M, N, K, pa.KS, pa.kc, pa.TPT); }
                    if (dq == 14) { const PlanAW pa = plan_aw(N, K, 2);
                                    return smallm_aw<2>(q, wp, wsc, dx, dy, M, N, K, pa.KS, pa.kc, pa.TPT); }
                    if (dq == 12) { const Plan5 p8 = plan5(N, K, 16);
                                    return smallm_v8<1, 16>(q, wp, wsc, dx, dy, M, N, K, p8.KS, p8.kc, p8.TPT); }
                    if (dq == 11) { const Plan5 p7 = plan5(N, K, 16);
                                    return smallm_v7<1, 16>(q, wp, wsc, dx, dy, M, N, K, p7.KS, p7.kc, p7.TPT); }
                    if (dq == 10) { const Plan5 p6 = plan5(N, K, 16);
                                    return smallm_v6<1, 16>(q, wp, wsc, dx, dy, M, N, K, p6.KS, p6.kc, p6.TPT); }
                    if (dq >= 7) { const int CT = dq == 7 ? 16 : dq == 8 ? 32 : 64;
                                   const Plan5 p5 = plan5(N, K, CT);
                                   if (CT == 16) return smallm_v5<1, 16>(q, wp, wsc, dx, dy, M, N, K, p5.KS, p5.kc, p5.TPT);
                                   if (CT == 32) return smallm_v5<1, 32>(q, wp, wsc, dx, dy, M, N, K, p5.KS, p5.kc, p5.TPT);
                                   return sycl::event{}; }
                    if (dq == 6) { const Plan4 p4 = plan4(N, K);
                                   return smallm_v4<1>(q, wp, wsc, dx, dy, M, N, K, p4.KS, p4.kc); }
                    if (dq == 4) e = smallm_v3<1, 16>(q, wp, wsc, dx, o, M, N, K, pl.KS, pl.kc);
                    else if (dq == 5) e = smallm_v3<1, 32>(q, wp, wsc, dx, o, M, N, K, pl.KS, pl.kc);
                    else if (dq == 3) e = M <= 8 ? smallm_v2<1>(q, wp, wsc, lut, dx, o, M, N, K, pl.KS, pl.kc)
                                            : smallm_v2<2>(q, wp, wsc, lut, dx, o, M, N, K, pl.KS, pl.kc);
                    else if (M <= 8) e = dq == 1 ? smallm<1, 1>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc)
                                  : dq == 2 ? smallm<1, 2>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc)
                                            : smallm<1, 0>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc);
                    else        e = dq == 1 ? smallm<2, 1>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc)
                                  : dq == 2 ? smallm<2, 2>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc)
                                            : smallm<2, 0>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc);
                    if (pl.KS > 1) e = reduce(q, part, dy, M, N, pl.KS, e);
                    return e;
                };
                ci = 0; launch().wait(); ci = 0;
                std::vector<float> hy(size_t(M) * N);
                q.memcpy(hy.data(), dy, hy.size() * 4).wait();
                // reference on a sample of columns (every column for small N)
                double maxrel = 0; int checked = 0;
                const int step = N > 16384 ? 97 : 1;
                for (int n = 0; n < N; n += step) {
                    for (int m = 0; m < M; ++m) {
                        double ref = 0, mag = 0;
                        for (int k = 0; k < K; ++k) {
                            const uint8_t byte = hp[size_t(n) * (K / 2) + k / 2];
                            const int nib = (k & 1) ? (byte >> 4) : (byte & 15);
                            const double w = double(e2m1_f(nib)) * std::ldexp(1.0, int(hs[size_t(n) * (K / 32) + k / 32]) - 127);
                            const double xv = double(float(hx[size_t(m) * K + k]));
                            ref += w * xv; mag += std::fabs(w * xv);
                        }
                        const double err = std::fabs(ref - double(hy[size_t(m) * N + n]));
                        maxrel = std::max(maxrel, mag > 0 ? err / mag : err);
                        ++checked;
                    }
                }
                const auto t0 = std::chrono::steady_clock::now();
                sycl::event e;
                for (int i = 0; i < iters; ++i) e = launch();
                e.wait();
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count() / iters;
                const double gbs = double(pb + sb) / (ms * 1e6);
                const bool ok = maxrel < 2e-4;
                all_ok = all_ok && ok;
                std::printf("%-16s N=%6d K=%5d M=%2d %s KS=%2d  %8.3f ms  %6.1f GB/s  maxrel %.2e %s\n",
                    sh.what, N, K, M, dq == 14 ? "AW2  " : dq == 13 ? "AW1  " : dq == 12 ? "v8   " : dq == 11 ? "v7   " : dq == 10 ? "v6   " : dq == 9 ? "v5c64" : dq == 8 ? "v5c32" : dq == 7 ? "v5c16" : dq == 6 ? "v4   " : "v3c16", dq >= 13 ? plan_aw(N, K, dq - 12).KS : dq == 6 ? plan4(N, K).KS : dq >= 7 ? plan5(N, K, dq == 8 ? 32 : 16).KS : pl.KS, ms, gbs, maxrel, ok ? "ok" : "FAIL");
            }
            sycl::free(part, q); sycl::free(dy, q); sycl::free(dx, q);
        }
        for (int c = 0; c < NC; ++c) { sycl::free(cps[c], q); sycl::free(css[c], q); }
    }
    std::printf("%s\n", all_ok ? "SMALLM PROBE PASS" : "SMALLM PROBE FAIL");
    return all_ok ? 0 : 1;
}
