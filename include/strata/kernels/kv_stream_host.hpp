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
inline int kv_mirror_resolve(KvStreamMirror& m, const int32_t* ids, const int32_t* steps, int n_q, int cap,
                             int page_size, int32_t* miss_block, int32_t* miss_slot) {
    const int64_t n = (int64_t) m.slot_block.size();
    const int32_t epoch = m.epoch + 1;
    int nmiss = 0;
    int64_t lookups = 0;
    std::vector<int32_t> local_list;
    for (int q = 0; q < n_q; ++q) {
        const int width = steps[(int64_t) q * kStepCount + kStepWidth];
        const int32_t* qi = ids + (int64_t) q * cap;
        // one thread's strided sweep, threads in order - the block scan's concatenation order
        for (int t = 0; t < kKvRT; ++t) {
            int local = 0;
            for (int i = t; i < width; i += kKvRT) {
                const int b = qi[i] / page_size;
                if (i > 0 && qi[i - 1] / page_size == b) continue;   // ids ascending: one lookup per block
                ++lookups;
                const int32_t sl = m.page_table[b];
                if (sl >= 0) {
                    m.slot_stamp[sl] = epoch;
                    m.slot_ref[sl] = 1;
                } else if (sl == -1) {
                    m.page_table[b] = -2;   // claimed once; a later query sees -2: neither hit nor new miss
                    if (local < kKvMaxL) local_list.push_back(b);
                    ++local;
                }
            }
            if (local > kKvMaxL) { m.overflow = true; local = kKvMaxL; }   // tripwire, host guard in resolve
            for (int k = 0; k < local; ++k) miss_block[nmiss + k] = local_list[k];
            nmiss += local;
            local_list.clear();
        }
    }
    // the CLOCK sweep, one step = kKvRT consecutive slots `(hand + thread) % n`, threads in order
    const int need = nmiss;
    int got = 0;
    int32_t hand = m.hand;
    std::vector<uint8_t> mine(kKvRT), cand(kKvRT);
    std::vector<int> rank(kKvRT);
    for (int scanned = 0; got < need && scanned < 3 * (int) n; scanned += kKvRT) {
        int total = 0;
        for (int t = 0; t < kKvRT; ++t) {
            const int32_t j = (int32_t) (((int64_t) hand + t) % n);
            mine[t] = (m.slot_stamp[j] == epoch) ? 1 : 0;
            cand[t] = (!mine[t] && (m.slot_block[j] < 0 || m.slot_ref[j] == 0)) ? 1 : 0;
            rank[t] = total;
            total += cand[t];
        }
        const int want = need - got;
        int cut = kKvRT;
        for (int t = 0; t < kKvRT; ++t)
            if (cand[t] && rank[t] == want - 1) { cut = t + 1; break; }
        for (int t = 0; t < kKvRT; ++t) {
            const int32_t j = (int32_t) (((int64_t) hand + t) % n);
            if (cand[t] && rank[t] < want) {
                miss_slot[got + rank[t]] = j;
                m.slot_stamp[j] = epoch;   // taken: a wrapping sweep must not take it twice
            } else if (t < cut && !mine[t]) {
                m.slot_ref[j] = 0;
            }
        }
        got += total < want ? total : want;
        hand = (int32_t) (((int64_t) hand + cut) % n);
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
