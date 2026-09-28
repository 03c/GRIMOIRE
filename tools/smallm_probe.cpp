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
        {5120, 5120, "o_proj / DN out"}, {12288, 5120, "q+gate"},
        {34816, 5120, "ffn gate_up"}, {5120, 17408, "ffn down"},
        {248320, 5120, "lm_head"}, {4096, 2048, "ornith q"}, {2048, 4096, "ornith o"}};
    bool all_ok = true;
    std::mt19937 rng(11);
    for (const auto& sh : shapes) {
        const int N = sh.N, K = sh.K;
        const size_t pb = size_t(N) * K / 2, sb = size_t(N) * K / 32;
        std::vector<uint8_t> hp(pb), hs(sb);
        for (auto& x : hp) x = uint8_t(rng());
        for (auto& x : hs) x = uint8_t(118 + rng() % 12);          // 2^-9 .. 2^2
        uint8_t* dp = sycl::malloc_device<uint8_t>(pb, q);
        uint8_t* ds = sycl::malloc_device<uint8_t>(sb, q);
        q.memcpy(dp, hp.data(), pb); q.memcpy(ds, hs.data(), sb).wait();
        for (int M : {1, 4, 8, 16}) {
            std::vector<bf16> hx(size_t(M) * K);
            std::uniform_real_distribution<float> U(-1.f, 1.f);
            for (auto& v : hx) v = bf16(U(rng));
            bf16* dx = sycl::malloc_device<bf16>(hx.size(), q);
            q.memcpy(dx, hx.data(), hx.size() * 2).wait();
            const Plan pl = plan_for(N, K);
            float* part = sycl::malloc_device<float>(size_t(pl.KS) * M * N, q);
            float* dy = sycl::malloc_device<float>(size_t(M) * N, q);
            for (int dq : {2, 3}) {
                auto launch = [&]() {
                    sycl::event e;
                    float* o = pl.KS > 1 ? part : dy;
                    if (dq == 3) e = M <= 8 ? smallm_v2<1>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc)
                                            : smallm_v2<2>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc);
                    else if (M <= 8) e = dq == 1 ? smallm<1, 1>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc)
                                  : dq == 2 ? smallm<1, 2>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc)
                                            : smallm<1, 0>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc);
                    else        e = dq == 1 ? smallm<2, 1>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc)
                                  : dq == 2 ? smallm<2, 2>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc)
                                            : smallm<2, 0>(q, dp, ds, lut, dx, o, M, N, K, pl.KS, pl.kc);
                    if (pl.KS > 1) e = reduce(q, part, dy, M, N, pl.KS, e);
                    return e;
                };
                launch().wait();
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
                    sh.what, N, K, M, dq == 1 ? "alu" : dq == 2 ? "slm" : dq == 3 ? "v2 " : "lut", pl.KS, ms, gbs, maxrel, ok ? "ok" : "FAIL");
            }
            sycl::free(part, q); sycl::free(dy, q); sycl::free(dx, q);
        }
        sycl::free(dp, q); sycl::free(ds, q);
    }
    std::printf("%s\n", all_ok ? "SMALLM PROBE PASS" : "SMALLM PROBE FAIL");
    return all_ok ? 0 : 1;
}
