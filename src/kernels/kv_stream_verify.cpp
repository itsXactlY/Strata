// src/kernels/kv_stream_verify.cpp - the host mirror (kv_stream_host.hpp) against the GPU map (kv_stream.cu).
//
// Prefetch contract: after every resolve the mirror equals the device map, and the next layer's victims
// predicted from the mirror are exactly what claim+resolve produce on both sides. Checks:
//   1. resolve without prefetch: after every call the mirror equals the device map (arrays + counters);
//   2. the prefetch path per layer: a speculative resolve on a mirror copy picks the pairs -> claim table
//      -> qsa_kv_fetch + qsa_kv_claim + device resolve, and mirror claim + mirror resolve -> maps equal;
//      the device resolve after a correct claim finds no new misses;
//   3. a drift tripwire: a pair for an already-resident block sets ctl[3] on the device and overflow on
//      the mirror.
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_stream_host.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

namespace k = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}
template <typename T> T* halloc(size_t n) {   // pinned, mapped; returns the device pointer
    void* h = nullptr;
    void* d = nullptr;
    ck(cudaHostAlloc(&h, n * sizeof(T) + 64, cudaHostAllocMapped), "hostalloc");
    std::memset(h, 0, n * sizeof(T) + 64);
    ck(cudaHostGetDevicePointer(&d, h, 0), "devptr");
    return (T*) d;
}

struct Pools {
    k::KvHostPools p;
    void alloc(int64_t pages, const k::QsaShapes& s, bool host) {
        const size_t rows = (size_t) pages * s.n_head_kv * s.page_size;
        p.k_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
        p.v_q = host ? halloc<int8_t>(rows * s.head_dim) : dalloc<int8_t>(rows * s.head_dim);
        p.k_scale = host ? halloc<uint16_t>(rows * (s.head_dim / k::KV_Q8_GROUP))
                         : dalloc<uint16_t>(rows * (s.head_dim / k::KV_Q8_GROUP));
        p.v_scale = host ? halloc<uint16_t>(rows * (s.head_dim / k::KV_Q8_GROUP))
                         : dalloc<uint16_t>(rows * (s.head_dim / k::KV_Q8_GROUP));
    }
    k::QsaAttnPools attn(const int32_t* table) const {
        k::QsaAttnPools a;
        a.k_q = p.k_q; a.v_q = p.v_q; a.k_scale = p.k_scale; a.v_scale = p.v_scale;
        a.page_table = table;
        return a;
    }
};

// selections shaped like the indexer's output
std::vector<int32_t> selection(int64_t n_kv, int64_t width, std::mt19937& rng) {
    std::vector<int32_t> ids;
    if (n_kv <= width) {
        for (int64_t c = 0; c < n_kv; ++c) ids.push_back((int32_t) c);
        return ids;
    }
    const int64_t n_bid = n_kv / 4, tail = n_kv - n_bid * 4;
    std::set<int64_t> blocks;
    const int64_t want_full = (width - tail) / 4, part = (width - tail) % 4;
    std::uniform_int_distribution<int64_t> any(0, n_bid - 1);
    while ((int64_t) blocks.size() < want_full + (part ? 1 : 0)) blocks.insert(any(rng));
    int64_t partial_block = part ? *std::next(blocks.begin(), (long) (rng() % blocks.size())) : -1;
    for (int64_t b : blocks) {
        const int64_t take = b == partial_block ? part : 4;
        for (int64_t i = 0; i < take; ++i) ids.push_back((int32_t) (b * 4 + i));
    }
    for (int64_t c = n_bid * 4; c < n_kv; ++c) ids.push_back((int32_t) c);
    std::sort(ids.begin(), ids.end());
    return ids;
}

bool maps_equal(const k::KvStreamMap& m, const k::KvStreamMirror& mm, int64_t nb, int64_t ns,
                const char* where) {
    std::vector<int32_t> pt(nb), sb(ns), st(ns), rf(ns);
    ck(cudaMemcpy(pt.data(), m.page_table, (size_t) nb * 4, cudaMemcpyDeviceToHost), "pt");
    ck(cudaMemcpy(sb.data(), m.slot_block, (size_t) ns * 4, cudaMemcpyDeviceToHost), "sb");
    ck(cudaMemcpy(st.data(), m.slot_stamp, (size_t) ns * 4, cudaMemcpyDeviceToHost), "st");
    ck(cudaMemcpy(rf.data(), m.slot_ref, (size_t) ns * 4, cudaMemcpyDeviceToHost), "rf");
    if (std::memcmp(pt.data(), mm.page_table.data(), (size_t) nb * 4) ||
        std::memcmp(sb.data(), mm.slot_block.data(), (size_t) ns * 4) ||
        std::memcmp(st.data(), mm.slot_stamp.data(), (size_t) ns * 4) ||
        std::memcmp(rf.data(), mm.slot_ref.data(), (size_t) ns * 4)) {
        std::fprintf(stderr, "mirror diverged: %s\n", where);
        return false;
    }
    return true;
}

bool run() {
    const int64_t NB = 2048, NS = 1024, NQ = 4, LAYERS = 6, WINDOWS = 40;   // NS >= resolve RT (1024); NB > NS forces eviction
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t cap = 40;
    const int page_size = (int) s.page_size;
    std::mt19937 rng(11);

    k::KvStreamMap m;
    m.page_table = dalloc<int32_t>(NB);
    m.slot_block = dalloc<int32_t>(NS); m.slot_stamp = dalloc<int32_t>(NS); m.slot_ref = dalloc<int32_t>(NS);
    m.miss_block = dalloc<int32_t>(NS); m.miss_slot = dalloc<int32_t>(NS);
    m.ctl = dalloc<int32_t>(k::kKvCtlInts);
    m.n_blocks = NB; m.n_slots = NS;
    k::kv_stream_reset(m, nullptr);

    k::KvStreamMirror mm;
    k::kv_mirror_reset(mm, NB, NS);

    void* claim_host = nullptr;
    void* claim_dev = nullptr;
    ck(cudaHostAlloc(&claim_host, (size_t) k::kKvClaimInts * 4, cudaHostAllocMapped), "claim hostalloc");
    std::memset(claim_host, 0, (size_t) k::kKvClaimInts * 4);
    ck(cudaHostGetDevicePointer(&claim_dev, claim_host, 0), "claim devptr");
    m.claim = (int32_t*) claim_dev;
    int32_t* claim_h = (int32_t*) claim_host;

    Pools slots, host;
    slots.alloc(NS, s, false);
    host.alloc(NB, s, true);

    int32_t* dev_ids = dalloc<int32_t>(LAYERS * NQ * cap);
    int32_t* dev_st = dalloc<int32_t>(LAYERS * NQ * k::kStepCount);
    std::vector<int32_t> miss_b(NS), miss_s(NS);   // host: the mirror writes these on the CPU

    uint64_t dev_misses_prefetch = 0, dev_lookups = 0, prefetch_rounds = 0, pairs_claimed = 0;

    for (int64_t w = 0; w < WINDOWS; ++w) {
        // flat host ids/steps per layer, same layout the device resolve gets
        std::vector<int32_t> flat_ids((size_t) (LAYERS * NQ * cap), 0), flat_st((size_t) (LAYERS * NQ * k::kStepCount), 0);
        for (int64_t L = 0; L < LAYERS; ++L) {
            for (int64_t q = 0; q < NQ; ++q) {
                const int64_t n_kv = 8 + (int64_t) (rng() % 200);
                const int64_t width = std::min<int64_t>(n_kv, cap);
                const std::vector<int32_t> sel = selection(n_kv, width, rng);
                std::copy(sel.begin(), sel.end(), flat_ids.begin() + (L * NQ + q) * cap);
                const int32_t st[k::kStepCount] = {(int32_t) (n_kv - 1), (int32_t) n_kv, (int32_t) (n_kv / 4),
                                                   (int32_t) width};
                std::copy(st, st + k::kStepCount, flat_st.begin() + (L * NQ + q) * k::kStepCount);
            }
        }
        ck(cudaMemcpy(dev_ids, flat_ids.data(), flat_ids.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(dev_st, flat_st.data(), flat_st.size() * 4, cudaMemcpyHostToDevice), "steps");

        auto layer_ids = [&](int64_t L) { return flat_ids.data() + L * NQ * cap; };
        auto layer_st = [&](int64_t L) { return flat_st.data() + L * NQ * k::kStepCount; };

        // layer 0: no prefetch (claim epoch 0). The mirror must see all NQ queries in ONE resolve,
        // like the device kernel: a single CLOCK pass over the combined miss list.
        claim_h[0] = 0; claim_h[1] = 0; claim_h[2] = 0;
        k::kv_stream_resolve(m, slots.attn(m.page_table), host.p, k::kKvInt8, dev_ids, dev_st, NQ, cap, s, nullptr);
        k::kv_mirror_resolve(mm, layer_ids(0), layer_st(0), NQ, (int) cap, page_size, miss_b.data(), miss_s.data());
        if (!maps_equal(m, mm, NB, NS, "after layer 0")) return false;

        for (int64_t L = 1; L < LAYERS; ++L) {
            // speculative resolve on a copy: the pairs the next resolve would place
            k::KvStreamMirror pred = mm;
            const int32_t placed =
                k::kv_mirror_resolve(pred, layer_ids(L), layer_st(L), NQ, (int) cap, page_size, miss_b.data(), miss_s.data());

            const int32_t epoch = mm.epoch, hand = mm.hand;
            if (placed > 0) {
                claim_h[0] = placed;
                claim_h[1] = epoch;
                claim_h[2] = hand;
                for (int32_t i = 0; i < placed; ++i) {
                    claim_h[3 + 2 * i] = miss_b[i];
                    claim_h[4 + 2 * i] = miss_s[i];
                }
            } else {
                claim_h[0] = 0; claim_h[1] = 0; claim_h[2] = 0;
            }

            // device: fetch + claim + resolve
            k::qsa_kv_fetch(m, slots.attn(m.page_table), host.p, k::kKvInt8, s, nullptr);
            k::qsa_kv_claim(m, nullptr);
            const uint64_t misses_before = k::kv_stream_counters(m).misses;
            k::kv_stream_resolve(m, slots.attn(m.page_table), host.p, k::kKvInt8,
                                 dev_ids + L * NQ * cap, dev_st + L * NQ * k::kStepCount, NQ, cap, s, nullptr);
            const k::KvStreamCounters dc = k::kv_stream_counters(m);
            const uint64_t layer_misses = dc.misses - misses_before;
            if (placed > 0) {
                dev_misses_prefetch += layer_misses;
                ++prefetch_rounds;
                pairs_claimed += (uint64_t) placed;
            }
            dev_lookups += dc.lookups;

            // mirror: claim + resolve (one batched call, same as the device)
            std::vector<int32_t> claim_copy((size_t) k::kKvClaimInts);
            std::memcpy(claim_copy.data(), claim_h, (size_t) k::kKvClaimInts * 4);
            k::kv_mirror_claim(mm, claim_copy.data());
            k::kv_mirror_resolve(mm, layer_ids(L), layer_st(L), NQ, (int) cap, page_size, miss_b.data(), miss_s.data());

            std::vector<int32_t> ctl(k::kKvCtlInts);
            ck(cudaMemcpy(ctl.data(), m.ctl, k::kKvCtlInts * 4, cudaMemcpyDeviceToHost), "ctl");
            if (ctl[3]) { std::fprintf(stderr, "claim tripwire at layer %lld\n", (long long) L); return false; }
            if (mm.overflow) { std::fprintf(stderr, "mirror overflow at layer %lld\n", (long long) L); return false; }
            if (!maps_equal(m, mm, NB, NS, "prefetch layer")) return false;
            // a correct claim must leave the device resolve with no new misses for those blocks
            if (placed > 0 && layer_misses > 0) {
                std::fprintf(stderr, "layer %lld: claimed %d pairs but resolve still missed %llu\n", (long long) L,
                             (int) placed, (unsigned long long) layer_misses);
                return false;
            }
            // counters must agree
            if (dc.lookups != mm.lookups || dc.misses != mm.misses) {
                std::fprintf(stderr, "counters differ at layer %lld: dev %llu/%llu vs mirror %llu/%llu\n", (long long) L,
                             (unsigned long long) dc.lookups, (unsigned long long) dc.misses,
                             (unsigned long long) mm.lookups, (unsigned long long) mm.misses);
                return false;
            }
        }
    }

    std::printf("  counters agree: %llu lookups, %llu prefetch rounds, %llu pairs claimed, %llu misses after claim\n",
                (unsigned long long) dev_lookups, (unsigned long long) prefetch_rounds,
                (unsigned long long) pairs_claimed, (unsigned long long) dev_misses_prefetch);
    if (prefetch_rounds == 0 || pairs_claimed == 0) {
        std::fprintf(stderr, "vacuous: no prefetch round claimed a pair (pool too large for this geometry)\n");
        return false;
    }

    // drift tripwire: a pair for an already-resident block must be caught on both sides
    {
        int32_t resident = -1;
        std::vector<int32_t> pt(NB);
        ck(cudaMemcpy(pt.data(), m.page_table, NB * 4, cudaMemcpyDeviceToHost), "pt");
        for (int64_t b = 0; b < NB; ++b)
            if (pt[b] >= 0) { resident = (int32_t) b; break; }
        if (resident < 0) { std::fprintf(stderr, "no resident block for the tripwire\n"); return false; }
        claim_h[0] = 1; claim_h[1] = mm.epoch; claim_h[2] = mm.hand;
        claim_h[3] = resident; claim_h[4] = 0;
        k::qsa_kv_claim(m, nullptr);
        std::vector<int32_t> ctl(k::kKvCtlInts);
        ck(cudaMemcpy(ctl.data(), m.ctl, k::kKvCtlInts * 4, cudaMemcpyDeviceToHost), "ctl");
        std::vector<int32_t> copy((size_t) k::kKvClaimInts);
        std::memcpy(copy.data(), claim_h, (size_t) k::kKvClaimInts * 4);
        const bool mir = k::kv_mirror_claim(mm, copy.data());
        if (ctl[3] != 1 || mm.overflow != 1 || !mir) {
            std::fprintf(stderr, "tripwire missed: dev ctl[3]=%d mirror overflow=%d\n", ctl[3], mm.overflow);
            return false;
        }
        std::printf("  drift tripwire: both sides caught the bogus pair\n");
    }
    return true;
}
}  // namespace

int main() {
    std::printf("kv_stream_verify: host mirror vs GPU map, prefetch contract\n");
    if (!run()) ++g_fail;
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
