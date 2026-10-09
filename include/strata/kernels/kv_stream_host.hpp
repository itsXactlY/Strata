// include/strata/kernels/kv_stream_host.hpp - the host mirror of resolve_kernel (docs/strata-tape-benefit.md).
//
// The verifier predicts the NEXT QSA layer's resolve on the CPU: it replays the walk, the CLOCK victim sweep and
// the table re-point over a host copy of the map, so it can claim the slots and issue the fetch copies on the
// copy stream while the current layer's attention still runs.  The replay must be BIT-IDENTICAL to resolve_kernel
// - same sweep order, same dedupe, same clock - or the device resolve and the mirror drift apart and the claim
// tripwires.  The invariant that keeps them together: the prefetch side owns the CLOCK state (epoch, hand,
// stamps), the in-graph claim_kernel adopts exactly the pairs the mirror picked, and the mirror replays claim
// before it replays the resolve that follows it.  The device map stays authoritative for attention; the mirror
// only ever predicts.
//
// The kernel's miss order is deterministic (per-thread lists concatenated by the block scan, and each block of
// one query is looked up by exactly one thread after the ascending-id dedupe), so replaying the RT threads in
// thread order reproduces it.  RT must match kv_stream.cu.
#pragma once

#include <cstdint>
#include <vector>

#include "strata/kernels/qsa.hpp"   // kStepCount / kStepWidth

namespace strata::kernels {

constexpr int kKvRT = 1024;   // the resolve block - must equal RT in kv_stream.cu
constexpr int kKvMaxL = 8;    // per-thread lookup list bound - must equal KV_MAXL in kv_stream.cu

struct KvStreamMirror {
    std::vector<int32_t> page_table, slot_block, slot_stamp, slot_ref;
    int32_t epoch = 0, hand = 0;
    bool overflow = false;
    uint64_t misses = 0, lookups = 0, calls = 0;
};

inline void kv_mirror_reset(KvStreamMirror& m, int64_t n_blocks, int64_t n_slots) {
    m.page_table.assign((size_t) n_blocks, -1);
    m.slot_block.assign((size_t) n_slots, -1);
    m.slot_stamp.assign((size_t) n_slots, -1);
    m.slot_ref.assign((size_t) n_slots, 0);
    m.epoch = 0;
    m.hand = 0;
    m.overflow = false;
    m.misses = m.lookups = m.calls = 0;
}

// Replay of resolve_kernel over the mirror.  ids/steps exactly as the kernel gets them (cells, not blocks);
// the pairs it picked land in `miss_block`/`miss_slot` (sized n_slots) and the counters land in the mirror.
// Returns the number of misses (ctl[2] of the replayed call).
// THE SCRATCH IS PER THREAD, NOT PER CALL.  The replay sits on the engine's critical path (once per layer per
// token, 48 times), and the obvious spelling - three vectors declared inside the sweep loop - allocates and
// frees kKvRT bytes per sweep step.  Measured at production shapes (cap 2048, 5120 slots, 16384 blocks):
// 33.9 us per layer-resolve = 1.63 ms of serial host work per token, before a single prefetch copy is issued.
//
// THE CONDITIONAL SUBTRACT AND THE SHIFT ARE THE OTHER HALF, and they are not micro-tuning.  `(hand + t) % n` and
// `qi[i] / page_size` are runtime divisors in the two innermost loops, 1024 times each per resolve, and those
// integer divisions cost more than everything else put together.  hand < n and t < RT <= n, so hand + t < 2n and
// one conditional subtract is exact; page_size is a power of two (one indexer block), so the shift is exact.
// Together with the hoisted scratch: 33.9 -> 14.5 us per layer-resolve, 1.63 -> 0.69 ms per token, and BIT-IDENTICAL
// to the previous spelling in all four map arrays, epoch, hand, misses, lookups and overflow.
inline int kv_mirror_resolve(KvStreamMirror& m, const int32_t* ids, const int32_t* steps, int n_q, int cap,
                             int page_size, int32_t* miss_block, int32_t* miss_slot) {
    struct Scratch {
        std::vector<int32_t> local;
        std::vector<uint8_t> mine, cand;
        std::vector<int32_t> rank;
    };
    static thread_local Scratch sc;
    if ((int) sc.local.size() < kKvMaxL) {
        sc.local.resize(kKvMaxL);
        sc.mine.resize(kKvRT);
        sc.cand.resize(kKvRT);
        sc.rank.resize(kKvRT);
    }
    int ps_shift = 0;
    while ((1 << ps_shift) < page_size) ++ps_shift;

    const int64_t n = (int64_t) m.slot_block.size();
    const int32_t epoch = m.epoch + 1;
    int nmiss = 0;
    int64_t lookups = 0;
    for (int q = 0; q < n_q; ++q) {
        const int width = steps[(int64_t) q * kStepCount + kStepWidth];
        const int32_t* qi = ids + (int64_t) q * cap;
        // one thread's strided sweep, threads in order - the block scan's concatenation order
        for (int t = 0; t < kKvRT; ++t) {
            int local = 0;
            for (int i = t; i < width; i += kKvRT) {
                const int b = qi[i] >> ps_shift;
                if (i > 0 && (qi[i - 1] >> ps_shift) == b) continue;   // ids ascending: one lookup per block
                ++lookups;
                const int32_t sl = m.page_table[b];
                if (sl >= 0) {
                    m.slot_stamp[sl] = epoch;
                    m.slot_ref[sl] = 1;
                } else if (sl == -1) {
                    m.page_table[b] = -2;   // claimed once; a later query sees -2: neither hit nor new miss
                    if (local < kKvMaxL) sc.local[local++] = b;
                }
            }
            if (local > kKvMaxL) { m.overflow = true; local = kKvMaxL; }   // tripwire, host guard in resolve
            for (int k = 0; k < local; ++k) miss_block[nmiss + k] = sc.local[k];
            nmiss += local;
        }
    }
    // the CLOCK sweep, one step = kKvRT consecutive slots `(hand + thread) % n`, threads in order
    const int need = nmiss;
    int got = 0;
    int32_t hand = m.hand;
    const int32_t nn = (int32_t) n;
    for (int scanned = 0; got < need && scanned < 3 * (int) n; scanned += kKvRT) {
        int total = 0;
        for (int t = 0; t < kKvRT; ++t) {
            int32_t j = hand + t;
            if (j >= nn) j -= nn;
            const uint8_t mi = (m.slot_stamp[j] == epoch) ? 1 : 0;
            const uint8_t cd = (!mi && (m.slot_block[j] < 0 || m.slot_ref[j] == 0)) ? 1 : 0;
            sc.mine[t] = mi;
            sc.cand[t] = cd;
            sc.rank[t] = total;
            total += cd;
        }
        const int want = need - got;
        int cut = kKvRT;
        for (int t = 0; t < kKvRT; ++t)
            if (sc.cand[t] && sc.rank[t] == want - 1) { cut = t + 1; break; }
        for (int t = 0; t < kKvRT; ++t) {
            int32_t j = hand + t;
            if (j >= nn) j -= nn;
            if (sc.cand[t] && sc.rank[t] < want) {
                miss_slot[got + sc.rank[t]] = j;
                m.slot_stamp[j] = epoch;   // taken: a wrapping sweep must not take it twice
            } else if (t < cut && !sc.mine[t]) {
                m.slot_ref[j] = 0;
            }
        }
        got += total < want ? total : want;
        hand += cut;
        if (hand >= nn) hand -= nn;
    }
    // re-point the table
    const int placed = got < need ? got : need;
    for (int k = 0; k < need; ++k) {
        const int b = miss_block[k];
        if (k >= placed) { m.page_table[b] = -1; continue; }
        const int32_t sl = miss_slot[k];
        const int old = m.slot_block[sl];
        if (old >= 0) m.page_table[old] = -1;
        m.slot_block[sl] = b;
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
        m.page_table[b] = sl;
    }
    m.epoch = epoch;
    m.hand = hand;
    m.misses += (uint64_t) placed;
    m.lookups += (uint64_t) lookups;
    m.calls += 1;
    if (placed < need) m.overflow = true;
    return placed;
}

// Replay of claim_kernel: adopt the pairs the mirror picked for the NEXT layer, before replaying that layer's
// resolve.  A pair whose block is already resident is mirror drift - the window rejects, same tripwire as the
// kernel's ctl[3].
inline bool kv_mirror_claim(KvStreamMirror& m, const int32_t* claim) {
    const int32_t epoch = claim[1];
    if (epoch == 0) return true;
    const int n = claim[0];
    for (int k = 0; k < n; ++k) {
        const int b = claim[3 + 2 * k];
        const int32_t sl = claim[4 + 2 * k];
        if (m.page_table[b] != -1) { m.overflow = true; continue; }
        const int old = m.slot_block[sl];
        if (old >= 0) m.page_table[old] = -1;
        m.slot_block[sl] = b;
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
        m.page_table[b] = sl;
    }
    m.epoch = epoch;
    m.hand = claim[2];
    return true;
}

}  // namespace strata::kernels
