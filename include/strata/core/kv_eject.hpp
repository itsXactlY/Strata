// include/strata/core/kv_eject.hpp - release a conversation's KV prefix back to the OS.
//
// WHY THIS EXISTS
// A rebase replaces the history under the live conversation. The KV cells of the
// old epoch are not merely unreachable, they are still *mapped*: qsa_state_zero
// zeroes the VRAM slots and resets the residency table, but the pinned host pools
// (st.host, cudaHostAlloc'd, device-mapped) keep every byte of the old epoch
// resident for the life of the session, and the parked ConversationBuffers hold
// theirs in ordinary pageable segments. Nothing on the path says "these cells
// belong to an epoch that no longer exists", so the pages stay faulted-in and
// the kernel treats them as reclaimable - which is how a dropped history ends up
// sitting in swap long after the messages that produced it were discarded.
//
// The eject is the missing half of the rebase. Daedalus knows the token position
// where the surviving tail begins; the engine needs to act on it, synchronously,
// so that the tail stands alone and the prefix is gone rather than merely
// unreferenced.
//
// WHAT IT DOES, AND WHY IT IS SAFE
// 1. Marks every page before the cut as non-resident. This is bookkeeping only -
//    the residency map already permits -1 and every later write to those cells
//    happens after this point, so no reader can name them (the same argument
//    qsa_state_zero makes about the unwritten tail).
// 2. Releases the pinned host cells with madvise(MADV_DONTNEED) so their pages
//    return to the allocator instead of staying faulted in. The next write to a
//    released cell faults them back in, which is the cost: one page fault per
//    cell the conversation actually reuses, paid once.
//    NOTE: on pinned pages madvise returns EINVAL on Linux. That is not a failure
//    to be swallowed - it is the signal that this region is genuinely resident,
//    and LockResult reports it rather than pretending the memory was freed.
// 3. Asks the platform to trim the process heap (malloc_trim) and drops the
//    arena's own page cache, so the freed bytes can leave rather than sit dirty.
//
// Correctness rests on one invariant: the cut is always at or before the current
// write position. A caller that passes a position ahead of the live cursor would
// discard cells that the current step still needs, so that is rejected outright
// rather than clamped.
#pragma once

#include <cstdint>
#include <string>

#include "strata/core/layer.hpp"

namespace strata::core {

/// What an eject actually achieved, so the caller can report it rather than assume.
struct EjectReport {
    bool        ok = false;
    int64_t     pages_ejected = 0;   ///< logical pages marked non-resident
    uint64_t    cells_ejected = 0;   ///< cells those pages cover
    uint64_t    pages_released = 0;  ///< pages whose host memory the allocator took back
    uint64_t    bytes_released = 0;  ///< host bytes returned to the allocator
    uint64_t    bytes_locked = 0;    ///< bytes that stayed resident (pinned): released no
    std::string note;                ///< why it stopped short, or what it did
};

/// Release the KV cells of the current conversation that precede `cut_cells`.
///
/// `cut_cells` is a cell position, not a token count: one cell per token per
/// layer, so the caller converts. Values at or beyond the session's written
/// position are rejected, because releasing live cells would corrupt the step
/// in flight.
///
/// Safe to call repeatedly, and safe to call on a state that is not streaming
/// (kv_mode 0): the residency map is the identity there, so there is nothing to
/// eject and the report says so rather than pretending work was done.
EjectReport kv_eject_before(const QsaState& st, const ModelGeometry& g,
                            int64_t cut_cells, int64_t written_cells, void* stream);

/// Release every parked conversation's host memory without dropping the entries
/// themselves, so the cache keeps its accounting but stops holding the bytes.
/// Used when the host is under pressure and the snapshots are the cheapest thing
/// to give back.
EjectReport kv_eject_all_parked(std::string& note);

}  // namespace strata::core