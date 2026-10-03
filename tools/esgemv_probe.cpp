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

// esgemv_probe.cpp -- decode (M=1) MXFP4 GEMV in ESIMD, Intel-llm-scaler style:
// one work-group per R output rows, KS threads splitting K, each thread
// streaming its rows' payload with contiguous 64-byte block loads, the
// activation from L1 (no SLM staging), E2M1 decoded through two 1 KB SLM
// tables (byte -> float of the low / high nibble), the E8M0 scale applied once
// per 32-element block.  Correctness against a CPU reference + GB/s on
// Ornith's decode shapes.
//
// Build (from the repo root):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 \
//     -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -cl-intel-256-GRF-per-thread" \
//     -O3 -std=c++20 tools/esgemv_probe.cpp -o bin/esgemv_probe
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

static float e2m1_f(int n) {
    static const float t[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = t[n & 7];
    return (n & 8) ? -v : v;
}

// y[n] = sum_k x[k] * W[n][k]; W MXFP4: payload [N][K/2] (low nibble = even k),
// scales [N][K/32] E8M0.  lut: [0,256) = value of the low nibble of byte b,
// [256,512) = value of the high nibble.
template <int R, int KS>
sycl::event esgemv(sycl::queue& q, const uint8_t* pay, const uint8_t* scl, const float* x,
                   float* y, int N, int K, const float* lut) {
    const int groups = (N + R - 1) / R;
    return q.parallel_for(sycl::nd_range<1>(size_t(groups) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<2048 + KS * R * 4>();
        const int t = int(it.get_local_id(0));
        for (int i = t; i < 16; i += KS)
            es::slm_block_store<float, 32>(i * 128, es::block_load<float, 32>(lut + i * 32));
        es::barrier();
        const int n0 = int(it.get_group(0)) * R;
        const int kp = K / KS, kb = t * kp;
        const int rowB = K / 2, rowS = K / 32;
        es::simd<float, 16> tot[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) tot[r] = 0.0f;
        for (int k = kb; k < kb + kp; k += 128) {
            // 128 elements: x (512 B, L1), per row 64 B of payload + 4 scale bytes
            es::simd<float, 128> xv = es::block_load<float, 128>(x + k);
            es::simd<uint8_t, 64> pb[R];
            es::simd<uint32_t, 4> sb[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const int n = n0 + r < N ? n0 + r : N - 1;
                pb[r] = es::block_load<uint8_t, 64>(pay + size_t(n) * rowB + k / 2);
                const uint8_t* sp = scl + size_t(n) * rowS + k / 32;
                es::simd<uint8_t, 4> s4 = es::block_load<uint8_t, 4>(sp);
                sb[r] = es::convert<uint32_t>(s4);
            }
            #pragma unroll
            for (int b = 0; b < 4; ++b) {              // 32-element MX block
                es::simd<float, 32> x32 = xv.template select<32, 1>(32 * b);
                es::simd<float, 16> xe = x32.template select<16, 2>(0);
                es::simd<float, 16> xo = x32.template select<16, 2>(1);
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    es::simd<uint8_t, 16> by = pb[r].template select<16, 1>(16 * b);
                    es::simd<uint32_t, 16> addr = es::convert<uint32_t>(by) << 2;
                    es::simd<float, 16> wl = es::slm_gather<float, 16>(addr);
                    es::simd<float, 16> wh = es::slm_gather<float, 16>(addr + 1024u);
                    es::simd<float, 16> part = wl * xe + wh * xo;
                    const uint32_t e = sb[r][b];
                    const float sc = sycl::bit_cast<float>(e ? e << 23 : 0x00400000u);
                    tot[r] += part * sc;
                }
            }
        }
        float res[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) res[r] = es::reduce<float>(tot[r], std::plus<>());
        if constexpr (KS == 1) {
            #pragma unroll
            for (int r = 0; r < R; ++r) if (n0 + r < N) y[n0 + r] = res[r];
        } else {
            es::simd<float, R> rv;
            #pragma unroll
            for (int r = 0; r < R; ++r) rv[r] = res[r];
            es::slm_block_store<float, R>(2048 + t * R * 4, rv);
            es::barrier();
            if (t == 0) {
                es::simd<float, R> acc = 0.0f;
                #pragma unroll
                for (int j = 0; j < KS; ++j) acc += es::slm_block_load<float, R>(2048 + j * R * 4);
                #pragma unroll
                for (int r = 0; r < R; ++r) if (n0 + r < N) y[n0 + r] = acc[r];
            }
        }
    });
}

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int iters = argc > 1 ? std::atoi(argv[1]) : 50;
    std::vector<float> hl(512);
    for (int b = 0; b < 256; ++b) { hl[b] = e2m1_f(b & 15); hl[256 + b] = e2m1_f(b >> 4); }
    float* lut = sycl::malloc_device<float>(512, q);
    q.memcpy(lut, hl.data(), 2048).wait();
    struct Shape { int N, K; const char* what; };
    const Shape shapes[] = {{8192, 2048, "la_qkv / q+gate"}, {4096, 2048, "z"},
                            {2048, 4096, "out / o"}, {1024, 2048, "k+v"}, {248320, 2048, "lm_head"}};
    std::mt19937 rng(5);
    bool all_ok = true;
    for (const auto& sh : shapes) {
        const int N = sh.N, K = sh.K;
        const size_t pb = size_t(N) * K / 2, sbz = size_t(N) * K / 32;
        std::vector<uint8_t> hp(pb), hs(sbz);
        for (auto& v : hp) v = uint8_t(rng());
        for (auto& v : hs) v = uint8_t(118 + rng() % 12);
        std::vector<float> hx(K);
        std::uniform_real_distribution<float> U(-1.f, 1.f);
        for (auto& v : hx) v = U(rng);
        uint8_t* dp = sycl::malloc_device<uint8_t>(pb, q);
        uint8_t* ds = sycl::malloc_device<uint8_t>(sbz, q);
        float* dx = sycl::malloc_device<float>(K, q);
        float* dy = sycl::malloc_device<float>(N, q);
        q.memcpy(dp, hp.data(), pb); q.memcpy(ds, hs.data(), sbz); q.memcpy(dx, hx.data(), K * 4).wait();
        // reference on a sample of rows
        std::vector<double> ref; std::vector<int> rows;
        for (int n = 0; n < N; n += (N > 16384 ? 97 : 7)) rows.push_back(n);
        for (int n : rows) {
            double s = 0;
            for (int k = 0; k < K; ++k) {
                const uint8_t byte = hp[size_t(n) * (K / 2) + k / 2];
                const int nib = (k & 1) ? (byte >> 4) : (byte & 15);
                s += double(e2m1_f(nib)) * std::ldexp(1.0, int(hs[size_t(n) * (K / 32) + k / 32]) - 127) * hx[k];
            }
            ref.push_back(s);
        }
        auto test = [&](const char* name, auto&& launch) {
            launch().wait();
            std::vector<float> hy(N);
            q.memcpy(hy.data(), dy, N * 4).wait();
            double maxrel = 0;
            for (size_t i = 0; i < rows.size(); ++i) {
                const double err = std::fabs(hy[rows[i]] - ref[i]) / (std::fabs(ref[i]) + 1e-3);
                maxrel = std::max(maxrel, err);
            }
            // back-to-back launches on the in-order queue, one wait: per-kernel
            // time including the inter-kernel gap, as inside the decode graph
            const auto a = std::chrono::steady_clock::now();
            sycl::event ev;
            for (int i = 0; i < iters; ++i) ev = launch();
            ev.wait();
            const double best = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - a).count() / iters;
            const bool ok = maxrel < 1e-3;
            all_ok = all_ok && ok;
            std::printf("  %-16s N=%6d K=%5d %-8s %8.2f us  %6.1f GB/s  maxrel %.1e %s\n", sh.what, N, K, name,
                        best, double(pb + sbz) / (best * 1e3), maxrel, ok ? "ok" : "FAIL");
        };
#define T(RR, KK) test("R" #RR "KS" #KK, [&]() { return esgemv<RR, KK>(q, dp, ds, dx, dy, N, K, lut); })
        T(2, 1); T(4, 1); T(8, 1); T(4, 2); T(4, 4); T(2, 4); T(8, 2); T(1, 8);
#undef T
        sycl::free(dp, q); sycl::free(ds, q); sycl::free(dx, q); sycl::free(dy, q);
    }
    std::printf("%s\n", all_ok ? "ESGEMV PROBE PASS" : "ESGEMV PROBE FAIL");
    return all_ok ? 0 : 1;
}
