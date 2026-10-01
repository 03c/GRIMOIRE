// decode_probe.cpp -- cold-weight decode kernel timing on one B70.
//
//   1. stream floor: how fast ANY single launch can read S bytes cold, for the
//      sizes decode actually moves per kernel (1-15 MB), vs thread count;
//   2. the engine's launch_gemv on Ornith's decode shapes, weights rotated
//      through enough copies that nothing is L2-resident (in-model condition);
//   3. the engine's fused-shared MoE gate_up / down with 8 random experts of
//      256 per call (cold, like a real token).
// Times are device time per kernel (profiling events) and back-to-back wall
// time per launch (in-order queue, includes the inter-kernel gap).
//
// Build (repo root, inside the container):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 \
//     -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -cl-intel-256-GRF-per-thread" \
//     -O3 -std=c++20 -fno-fast-math -ffp-contract=fast -fno-math-errno -I include -I src \
//     tools/decode_probe.cpp src/gemv_decode.cpp src/moe_kernels.cpp -o bin/decode_probe
#include "kernels.hpp"
#include "b70/moe.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace b70;
namespace es = sycl::ext::intel::esimd;

static sycl::queue* gq = nullptr;

struct Timing { double dev_us, wall_us; };

// Run `launch(i)` n times back to back; device time from profiling events,
// wall time from one wait at the end.
static Timing time_it(int n, const std::function<sycl::event(int)>& launch) {
    sycl::queue& q = *gq;
    for (int i = 0; i < 3; ++i) launch(i);           // warm-up (code, TLB)
    q.wait();
    std::vector<sycl::event> evs;
    evs.reserve(n);
    const auto a = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) evs.push_back(launch(i + 3));
    q.wait();
    const double wall = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - a).count() / n;
    double dev = 0;
    for (auto& e : evs) {
        const uint64_t s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t f = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        dev += double(f - s) * 1e-3;
    }
    return {dev / n, wall};
}

// Pure streaming read: `threads` ESIMD threads, each reads `per` contiguous
// bytes with 64-byte block loads, U of them issued before use.
template <int U>
static sycl::event stream_read(const uint8_t* base, size_t per, int threads, uint32_t* sink) {
    return gq->parallel_for(sycl::nd_range<1>(size_t(threads), 1),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const size_t t = it.get_global_id(0);
            const uint32_t* p = reinterpret_cast<const uint32_t*>(base + t * per);
            es::simd<uint32_t, 16> acc = 0;
            for (size_t o = 0; o < per / 4; o += 16 * U) {
                es::simd<uint32_t, 16> v[U];
                #pragma unroll
                for (int u = 0; u < U; ++u) v[u] = es::block_load<uint32_t, 16>(p + o + 16 * u);
                #pragma unroll
                for (int u = 0; u < U; ++u) acc += v[u];
            }
            const uint32_t r = es::reduce<uint32_t>(acc, std::plus<>());
            if (r == 0x9e3779b9u) sink[0] = r;
        });
}

// ---------------------------------------------------------------------------
// ESIMD MoE decode kernels (candidates).  Intel llm-scaler shape: a thread
// streams whole weight rows with contiguous 64-byte block loads; its slice of
// the activation stays in registers, split into even / odd K (the low / high
// nibble of each payload byte); E2M1 is decoded in the ALU: the nibble's
// three magnitude bits placed at fp16 bits 9..11 with the sign at bit 15 ARE
// the fp16 value times 2^-14 (fp16 subnormals hold the E2M1 subnormal), and
// the 2^14 folds into the E8M0 block scale.  No SLM table, no barrier until
// the K-split reduction.
// ---------------------------------------------------------------------------
template <int N>
SYCL_ESIMD_FUNCTION inline void e2m1_split(es::simd<uint8_t, N> b, es::simd<float, N>& lo,
                                           es::simd<float, N>& hi) {
    es::simd<uint16_t, N> u = b;
    es::simd<uint16_t, N> l = ((u & 0x7) << 9) | ((u & 0x8) << 12);
    es::simd<uint16_t, N> h = ((u & 0x70) << 5) | ((u & 0x80) << 8);
    es::simd<sycl::half, N> lh = l.template bit_cast_view<sycl::half>();
    es::simd<sycl::half, N> hh = h.template bit_cast_view<sycl::half>();
    lo = lh;
    hi = hh;
}
SYCL_ESIMD_FUNCTION inline float e8m0_x2p14(uint32_t e) {   // 2^(e-127) * 2^14
    return sycl::bit_cast<float>((e + 14u) << 23);
}

// One 128-element step of one row: acc += sum_b scale_b * (lo.xe + hi.xo).
template <int KP>
SYCL_ESIMD_FUNCTION inline void row_step(const uint8_t* prow, const uint8_t* srow, int st,
                                         es::simd<float, KP / 2>& xe,
                                         es::simd<float, KP / 2>& xo,
                                         es::simd<float, 16>& acc) {
    es::simd<uint8_t, 64> pb = es::block_load<uint8_t, 64>(prow + st * 64);
    es::simd<uint8_t, 4> sb = es::block_load<uint8_t, 4>(srow + st * 4);
    es::simd<float, 64> wl, wh;
    e2m1_split<64>(pb, wl, wh);
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        es::simd<float, 16> part = wl.template select<16, 1>(16 * b) * xe.template select<16, 1>(st * 64 + 16 * b)
                                 + wh.template select<16, 1>(16 * b) * xo.template select<16, 1>(st * 64 + 16 * b);
        acc += part * e8m0_x2p14(uint32_t(sb[b]));
    }
}

// gate_up + SiLU*up for top-k routed experts plus the shared expert (slot TK)
// and the shared-expert gate (one extra work-group).  Work-group = KS threads
// splitting H; a thread owns R (gate, up) row pairs of one slot.
template <int R, int KS, int H>
sycl::event moe_gu_es(sycl::queue& q, QuantWeight w, QuantWeight ws, const uint16_t* gate_w,
                      const int32_t* d_expert, const float* x, float* h, float* gate_out,
                      int I, int TK) {
    constexpr int KP = H / KS;
    static_assert(KP % 128 == 0, "K slice must be whole 128-element steps");
    const int rg = I / R;
    const int n_wg = (TK + 1) * rg + (gate_w ? 1 : 0);
    const uint8_t* wp = w.payload; const uint8_t* wsc = static_cast<const uint8_t*>(w.scales);
    const uint8_t* sp = ws.payload; const uint8_t* ssc = static_cast<const uint8_t*>(ws.scales);
    const int64_t wrb = w.row_bytes, wrs = w.row_scales, srb = ws.row_bytes, srs = ws.row_scales;
    return q.parallel_for(sycl::nd_range<1>(size_t(n_wg) * KS, KS), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<(KS * R * 2 * 4 > KS * 16 ? KS * R * 2 * 4 : KS * 16)>();
        const int t = int(it.get_local_id(0));
        const int g = int(it.get_group(0));
        const int kb = t * KP;
        if (g == (TK + 1) * rg) {                           // shared-expert gate
            es::simd<float, 16> a = 0.0f;
            for (int k = kb; k < kb + KP; k += 16) {
                es::simd<uint16_t, 16> gb = es::block_load<uint16_t, 16>(gate_w + k);
                es::simd<uint32_t, 16> gu = es::convert<uint32_t>(gb) << 16;
                es::simd<float, 16> gf = gu.template bit_cast_view<float>();
                a += gf * es::block_load<float, 16>(x + k);
            }
            es::simd<float, 4> av = 0.0f;
            av[0] = es::reduce<float>(a, std::plus<>());
            es::slm_block_store<float, 4>(t * 16, av);
            es::barrier();
            if (t == 0) {
                float s = 0.0f;
                es::simd<float, 4 * KS> p = es::slm_block_load<float, 4 * KS>(0);
                #pragma unroll
                for (int j = 0; j < KS; ++j) s += p[4 * j];
                gate_out[0] = 1.0f / (1.0f + sycl::exp(-s));
            }
            return;
        }
        const int slot = g / rg, i0 = (g % rg) * R;
        const bool sh = slot == TK;
        const uint8_t* pay = sh ? sp : wp;
        const uint8_t* scl = sh ? ssc : wsc;
        const int64_t rb = sh ? srb : wrb, rs = sh ? srs : wrs;
        const int64_t row0 = sh ? int64_t(i0) : int64_t(d_expert[slot]) * 2 * I + i0;
        es::simd<float, KP> xs = es::block_load<float, KP>(x + kb);
        es::simd<float, KP / 2> xe = xs.template select<KP / 2, 2>(0);
        es::simd<float, KP / 2> xo = xs.template select<KP / 2, 2>(1);
        es::simd<float, 16> ga[R], ua[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) { ga[r] = 0.0f; ua[r] = 0.0f; }
        #pragma unroll 1
        for (int st = 0; st < KP / 128; ++st) {
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const int64_t gr = row0 + r, ur = row0 + I + r;
                row_step<KP>(pay + gr * rb + kb / 2, scl + gr * rs + kb / 32, st, xe, xo, ga[r]);
                row_step<KP>(pay + ur * rb + kb / 2, scl + ur * rs + kb / 32, st, xe, xo, ua[r]);
            }
        }
        es::simd<float, 2 * R> red;
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            red[2 * r] = es::reduce<float>(ga[r], std::plus<>());
            red[2 * r + 1] = es::reduce<float>(ua[r], std::plus<>());
        }
        if constexpr (KS > 1) {
            es::slm_block_store<float, 2 * R>(t * 2 * R * 4, red);
            es::barrier();
            if (t != 0) return;
            red = 0.0f;
            #pragma unroll
            for (int j = 0; j < KS; ++j) red += es::slm_block_load<float, 2 * R>(j * 2 * R * 4);
        }
        es::simd<float, R> gg = red.template select<R, 2>(0), uu = red.template select<R, 2>(1);
        es::simd<float, R> o = gg / (1.0f + es::exp(-gg)) * uu;
        es::block_store<float, R>(h + int64_t(slot) * I + i0, o);
    });
}

// down + routed-weight reduction across the TK experts and the shared expert.
// Work-group = TK+1 threads, thread s = slot s; a work-group owns R output
// rows; slot partials meet in SLM and thread 0 sums them in slot order.
template <int R, int I>
sycl::event moe_dn_es(sycl::queue& q, QuantWeight w, QuantWeight wd, const int32_t* d_expert,
                      const float* d_weight, const float* gate_in, const float* h, float* y,
                      int H, int TK) {
    static_assert(I % 128 == 0, "");
    const uint8_t* wp = w.payload; const uint8_t* wsc = static_cast<const uint8_t*>(w.scales);
    const uint8_t* sp = wd.payload; const uint8_t* ssc = static_cast<const uint8_t*>(wd.scales);
    const int64_t wrb = w.row_bytes, wrs = w.row_scales, srb = wd.row_bytes, srs = wd.row_scales;
    const int S = TK + 1;
    return q.parallel_for(sycl::nd_range<1>(size_t(H / R) * S, S), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        es::slm_init<16 * R * 4>();
        const int s = int(it.get_local_id(0));
        const int o0 = int(it.get_group(0)) * R;
        const bool sh = s == TK;
        const uint8_t* pay = sh ? sp : wp;
        const uint8_t* scl = sh ? ssc : wsc;
        const int64_t rb = sh ? srb : wrb, rs = sh ? srs : wrs;
        const int64_t row0 = sh ? int64_t(o0) : int64_t(d_expert[s]) * H + o0;
        const float wt = sh ? (gate_in ? gate_in[0] : 1.0f) : d_weight[s];
        es::simd<float, I> hs = es::block_load<float, I>(h + int64_t(s) * I);
        es::simd<float, I / 2> he = hs.template select<I / 2, 2>(0);
        es::simd<float, I / 2> ho = hs.template select<I / 2, 2>(1);
        es::simd<float, 16> acc[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = 0.0f;
        #pragma unroll
        for (int st = 0; st < I / 128; ++st) {
            #pragma unroll
            for (int r = 0; r < R; ++r)
                row_step<I>(pay + (row0 + r) * rb, scl + (row0 + r) * rs, st, he, ho, acc[r]);
        }
        es::simd<float, R> part;
        #pragma unroll
        for (int r = 0; r < R; ++r) part[r] = es::reduce<float>(acc[r], std::plus<>()) * wt;
        es::slm_block_store<float, R>(s * R * 4, part);
        es::barrier();
        if (s != 0) return;
        es::simd<float, R> tot = 0.0f;
        for (int j = 0; j < S; ++j) tot += es::slm_block_load<float, R>(j * R * 4);
        es::block_store<float, R>(y + o0, tot);
    });
}

static void fill_mxfp4(uint8_t* pay, uint8_t* scl, size_t pb, size_t sb, uint32_t seed) {
    // device-side fill: random nibbles, E8M0 scales 118..129
    gq->parallel_for(sycl::range<1>(pb), [=](sycl::id<1> i) {
        uint32_t h = uint32_t(i[0]) * 2654435761u ^ seed;
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        pay[i] = uint8_t(h);
    });
    gq->parallel_for(sycl::range<1>(sb), [=](sycl::id<1> i) {
        uint32_t h = uint32_t(i[0]) * 2246822519u ^ (seed * 7u);
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        scl[i] = uint8_t(118 + h % 12);
    });
    gq->wait();
}

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v,
                  {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
    gq = &q;
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const std::string what = argc > 1 ? argv[1] : "all";
    const int NIT = 200;

    uint32_t* sink = sycl::malloc_device<uint32_t>(16, q);

    if (what == "all" || what == "stream") {
        const size_t POOL = size_t(1) << 30;           // 1 GiB, rotate through it
        uint8_t* pool = sycl::malloc_device<uint8_t>(POOL, q);
        q.memset(pool, 1, POOL).wait();
        std::printf("== stream floor (cold, rotating over 1 GiB)\n");
        const double sizes_mb[] = {1.11, 4.46, 8.91, 10.0, 15.0};
        for (double mb : sizes_mb) {
            for (int threads : {512, 1024, 2048, 4096, 8192}) {
                for (int U : {2, 4, 8}) {
                    size_t per = size_t(mb * 1e6 / threads);
                    per = (per + 64 * U - 1) / (64 * U) * (64 * U);
                    const size_t bytes = per * threads;
                    const size_t slots = POOL / bytes;
                    auto L = [&](int i) -> sycl::event {
                        const uint8_t* b = pool + (size_t(i) * 7919 % slots) * bytes;
                        if (U == 2) return stream_read<2>(b, per, threads, sink);
                        if (U == 4) return stream_read<4>(b, per, threads, sink);
                        return stream_read<8>(b, per, threads, sink);
                    };
                    const Timing t = time_it(NIT, L);
                    std::printf("  %6.2f MB  thr %5d  U%d  per %6zu B  dev %7.2f us (%5.0f GB/s)  wall %7.2f us\n",
                                bytes / 1e6, threads, U, per, t.dev_us, bytes / (t.dev_us * 1e3), t.wall_us);
                }
            }
        }
        sycl::free(pool, q);
    }

    if (what == "all" || what == "gemv") {
        struct Shape { int N, K; const char* name; };
        const Shape shapes[] = {{8192, 2048, "la_qkv/q"}, {4096, 2048, "z"}, {2048, 4096, "out/o"},
                                {1024, 2048, "k+v"}, {12352, 2048, "la_all"}};
        float* x = sycl::malloc_device<float>(8192, q);
        float* y = sycl::malloc_device<float>(1 << 18, q);
        q.fill(x, 0.01f, 8192).wait();
        std::printf("== engine launch_gemv, MXFP4, cold (rotating copies, >= 512 MB)\n");
        for (const auto& s : shapes) {
            const size_t pb = size_t(s.N) * s.K / 2, sb = size_t(s.N) * s.K / 32;
            const int copies = int(std::max<size_t>(8, (size_t(512) << 20) / (pb + sb)));
            uint8_t* pay = sycl::malloc_device<uint8_t>(pb * copies, q);
            uint8_t* scl = sycl::malloc_device<uint8_t>(sb * copies, q);
            fill_mxfp4(pay, scl, pb * copies, sb * copies, 17);
            auto L = [&](int i) -> sycl::event {
                const int c = (i * 7) % copies;
                QuantWeight w; w.fmt = Fmt::MXFP4; w.N = s.N; w.K = s.K;
                w.payload = pay + size_t(c) * pb; w.scales = scl + size_t(c) * sb;
                w.row_bytes = s.K / 2; w.row_scales = s.K / 32;
                return launch_gemv(q, w, x, y, {});
            };
            const Timing t = time_it(NIT, L);
            std::printf("  %-9s N=%6d K=%5d  %6.2f MB  dev %7.2f us (%5.0f GB/s)  wall %7.2f us\n",
                        s.name, s.N, s.K, (pb + sb) / 1e6, t.dev_us, (pb + sb) / (t.dev_us * 1e3), t.wall_us);
            sycl::free(pay, q); sycl::free(scl, q);
        }
        sycl::free(x, q); sycl::free(y, q);
    }

    if (what == "all" || what == "moe") {
        const int H = 2048, I = 512, E = 256, TK = 8;
        MoeLayer L;
        L.cfg.hidden = H; L.cfg.inter = I; L.cfg.num_experts = E; L.cfg.top_k = TK; L.cfg.shared_inter = I;
        const size_t gpb = size_t(E) * 2 * I * H / 2, gsb = size_t(E) * 2 * I * H / 32;
        const size_t dpb = size_t(E) * H * I / 2, dsb = size_t(E) * H * I / 32;
        uint8_t* gp = sycl::malloc_device<uint8_t>(gpb, q); uint8_t* gs = sycl::malloc_device<uint8_t>(gsb, q);
        uint8_t* dp = sycl::malloc_device<uint8_t>(dpb, q); uint8_t* ds = sycl::malloc_device<uint8_t>(dsb, q);
        fill_mxfp4(gp, gs, gpb, gsb, 3); fill_mxfp4(dp, ds, dpb, dsb, 5);
        L.gate_up.fmt = Fmt::MXFP4; L.gate_up.N = E * 2 * I; L.gate_up.K = H;
        L.gate_up.payload = gp; L.gate_up.scales = gs; L.gate_up.row_bytes = H / 2; L.gate_up.row_scales = H / 32;
        L.down.fmt = Fmt::MXFP4; L.down.N = E * H; L.down.K = I;
        L.down.payload = dp; L.down.scales = ds; L.down.row_bytes = I / 2; L.down.row_scales = I / 32;
        // shared expert: its own [2I][H] and [H][I]
        const size_t sgpb = size_t(2) * I * H / 2, sgsb = size_t(2) * I * H / 32;
        const size_t sdpb = size_t(H) * I / 2, sdsb = size_t(H) * I / 32;
        uint8_t* sgp = sycl::malloc_device<uint8_t>(sgpb, q); uint8_t* sgs = sycl::malloc_device<uint8_t>(sgsb, q);
        uint8_t* sdp = sycl::malloc_device<uint8_t>(sdpb, q); uint8_t* sds = sycl::malloc_device<uint8_t>(sdsb, q);
        fill_mxfp4(sgp, sgs, sgpb, sgsb, 9); fill_mxfp4(sdp, sds, sdpb, sdsb, 11);
        QuantWeight sgu; sgu.fmt = Fmt::MXFP4; sgu.N = 2 * I; sgu.K = H; sgu.payload = sgp; sgu.scales = sgs;
        sgu.row_bytes = H / 2; sgu.row_scales = H / 32;
        QuantWeight sdn; sdn.fmt = Fmt::MXFP4; sdn.N = H; sdn.K = I; sdn.payload = sdp; sdn.scales = sds;
        sdn.row_bytes = I / 2; sdn.row_scales = I / 32;
        uint16_t* gw = sycl::malloc_device<uint16_t>(H, q);
        q.fill(gw, uint16_t(0x3c00), H).wait();          // bf16 ~0.0117
        // routing tables: NIT+8 tokens, 8 distinct random experts each
        const int NT = NIT + 8;
        std::vector<int32_t> he(size_t(NT) * TK); std::vector<float> hw(size_t(NT) * TK);
        std::mt19937 rng(1234);
        for (int t = 0; t < NT; ++t) {
            std::vector<int> pick;
            while (int(pick.size()) < TK) {
                const int e = int(rng() % E);
                if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
            }
            for (int k = 0; k < TK; ++k) { he[size_t(t) * TK + k] = pick[k]; hw[size_t(t) * TK + k] = 1.0f / TK; }
        }
        int32_t* de = sycl::malloc_device<int32_t>(he.size(), q);
        float* dw = sycl::malloc_device<float>(hw.size(), q);
        q.memcpy(de, he.data(), he.size() * 4); q.memcpy(dw, hw.data(), hw.size() * 4).wait();
        float* x = sycl::malloc_device<float>(H, q);
        float* h = sycl::malloc_device<float>(size_t(TK + 1) * I, q);
        float* y = sycl::malloc_device<float>(H, q);
        float* go = sycl::malloc_device<float>(4, q);
        {
            std::vector<float> hx(H);
            std::uniform_real_distribution<float> U(-1.0f, 1.0f);
            for (auto& v : hx) v = U(rng);
            q.memcpy(x, hx.data(), H * 4).wait();
            std::vector<uint16_t> hg(H);
            for (auto& v : hg) v = uint16_t(sycl::bit_cast<uint32_t>(U(rng) * 0.05f) >> 16);
            q.memcpy(gw, hg.data(), H * 2).wait();
        }
        const double gub = double(TK + 1) * (2.0 * I * H / 2 + 2.0 * I * H / 32);
        const double dnb = double(TK + 1) * (double(H) * I / 2 + double(H) * I / 32);
        std::printf("== engine fused-shared MoE (8 random of 256 experts + shared, cold)\n");
        Timing tg = time_it(NIT, [&](int i) {
            return launch_moe_gate_up_shared(q, L, sgu, gw, de + size_t(i % NT) * TK, x, h, go, {}); });
        std::printf("  gate_up  %6.2f MB  dev %7.2f us (%5.0f GB/s)  wall %7.2f us\n",
                    gub / 1e6, tg.dev_us, gub / (tg.dev_us * 1e3), tg.wall_us);
        Timing td = time_it(NIT, [&](int i) {
            return launch_moe_down_shared(q, L, sdn, de + size_t(i % NT) * TK, dw + size_t(i % NT) * TK,
                                          go, h, y, {}); });
        std::printf("  down     %6.2f MB  dev %7.2f us (%5.0f GB/s)  wall %7.2f us\n",
                    dnb / 1e6, td.dev_us, dnb / (td.dev_us * 1e3), td.wall_us);

        // ---- candidates: correctness against the engine, then cold timing
        const int t0 = 5;                                    // one fixed routing for the checks
        float* h2 = sycl::malloc_device<float>(size_t(TK + 1) * I, q);
        float* y2 = sycl::malloc_device<float>(H, q);
        float* go2 = sycl::malloc_device<float>(4, q);
        launch_moe_gate_up_shared(q, L, sgu, gw, de + t0 * TK, x, h, go, {});
        launch_moe_down_shared(q, L, sdn, de + t0 * TK, dw + t0 * TK, go, h, y, {}).wait();
        std::vector<float> rh(size_t(TK + 1) * I), ry(H), rg(1);
        q.memcpy(rh.data(), h, rh.size() * 4); q.memcpy(ry.data(), y, H * 4); q.memcpy(rg.data(), go, 4).wait();
        auto maxrel = [](const std::vector<float>& a, const std::vector<float>& b) {
            double m = 0, scale = 0;
            for (float v : a) scale = std::max(scale, double(std::fabs(v)));
            for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(double(a[i]) - b[i]) / (scale + 1e-20));
            return m;
        };
        auto gu_case = [&](const char* nm, auto launch) {
            launch(t0).wait();
            std::vector<float> th(rh.size()), tg(1);
            q.memcpy(th.data(), h2, th.size() * 4); q.memcpy(tg.data(), go2, 4).wait();
            const double e = maxrel(rh, th), eg = std::fabs(tg[0] - rg[0]);
            const Timing t = time_it(NIT, [&](int i) { return launch(i % NT); });
            std::printf("  gate_up %-10s dev %7.2f us (%5.0f GB/s)  wall %7.2f us  err %.1e gate %.1e %s\n", nm,
                        t.dev_us, gub / (t.dev_us * 1e3), t.wall_us, e, eg, (e < 1e-4 && eg < 1e-5) ? "ok" : "FAIL");
        };
#define GU(RR, KK) gu_case("R" #RR " KS" #KK, [&](int i) { \
            return moe_gu_es<RR, KK, 2048>(q, L.gate_up, sgu, gw, de + size_t(i) * TK, x, h2, go2, I, TK); })
        GU(2, 4); GU(4, 4); GU(8, 4); GU(4, 2); GU(8, 2); GU(4, 8); GU(2, 8);
#undef GU
        auto dn_case = [&](const char* nm, auto launch) {
            launch(t0).wait();
            std::vector<float> ty(H);
            q.memcpy(ty.data(), y2, H * 4).wait();
            const double e = maxrel(ry, ty);
            const Timing t = time_it(NIT, [&](int i) { return launch(i % NT); });
            std::printf("  down    %-10s dev %7.2f us (%5.0f GB/s)  wall %7.2f us  err %.1e %s\n", nm,
                        t.dev_us, dnb / (t.dev_us * 1e3), t.wall_us, e, e < 1e-4 ? "ok" : "FAIL");
        };
        // the engine's h / gate for routing t0 feed every down candidate
#define DN(RR) dn_case("R" #RR, [&](int i) { \
            return moe_dn_es<RR, 512>(q, L.down, sdn, de + size_t(i) * TK, dw + size_t(i) * TK, go, h, y2, H, TK); })
        DN(2); DN(4); DN(8);
#undef DN
    }
    std::printf("DECODE PROBE DONE\n");
    return 0;
}
