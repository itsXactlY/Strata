// src/core/kv_eject.cpp - see include/strata/core/kv_eject.hpp for the contract.
//
// The host pools are one cudaHostAlloc region carved into k/v (+scales). Cells
// are indexed identically across every pointer in the pool, so a cell range maps
// to a byte range in each: cell c sits at offset c * kv_block_bytes(s, fmt).
// That is the same arithmetic kv_stream_resolve uses when it copies a block back
// in, which is why releasing a range here and faulting it back there agree.

#include "strata/core/kv_eject.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#if !defined(_WIN32)
#include <malloc.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_stream.hpp"

namespace strata::core {
namespace {

#if !defined(_WIN32)
/// Release [p, p + bytes) back to the allocator. Returns the bytes actually
/// released: madvise on a page-locked region fails with EINVAL, and that is
/// reported as "stayed resident" rather than hidden.
uint64_t release_range(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return 0;
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t begin = (uintptr_t) p & ~(page - 1);
    uintptr_t end = ((uintptr_t) p + bytes + page - 1) & ~(page - 1);
    if (madvise((void*) begin, (uintptr_t) (end - begin), MADV_DONTNEED) != 0) return 0;
    return (uint64_t) (end - begin);
}
#else
uint64_t release_range(void*, uint64_t) { return 0; }
#endif

/// Trim the allocator's free lists so freed arenas can leave the process. glibc
/// keeps a top-of-heap chunk and per-arena free lists; without this the pages are
/// free but still resident, and the kernel swaps them rather than reclaiming them.
void trim_heap() {
#if !defined(_WIN32)
    malloc_trim(0);
#endif
}

}  // namespace

EjectReport kv_eject_before(const QsaState& st, const ModelGeometry& g,
                            int64_t cut_cells, int64_t written_cells, void* stream) {
    EjectReport rep;
    // qsa_shapes() is file-static in layer.cpp (an anonymous namespace), so the
    // geometry is resolved here from the same two sources it uses: the model's
    // real shape, then the per-model overrides. Duplicating four assignments is
    // better than exporting a symbol whose only other user would be this file.
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;

    if (cut_cells <= 0) {
        rep.ok = true;
        rep.note = "cut at or before the start: nothing to eject";
        return rep;
    }
    if (written_cells > 0 && cut_cells > written_cells) {
        // Clamping here would silently discard live cells; the caller has a bug
        // and the loop below would fault mid-step. Refuse instead.
        rep.note = "refused: cut is ahead of the written position";
        return rep;
    }
    if (st.kv_mode != 1) {
        // Mode 0 is the identity map with no host copy: every cell is its own
        // page and there is nothing streamed to release.
        rep.ok = true;
        rep.note = "state is not streamed (kv_mode 0): no host copy to release";
        return rep;
    }
    if (s.page_size <= 0 || st.n_pages <= 0) {
        rep.note = "no residency map";
        return rep;
    }

    const int64_t cut_pages = std::min<int64_t>(cut_cells / s.page_size, st.n_pages);

    // 1. Retire the pages ON THE DEVICE. The page table lives in VRAM (layer.cpp
    //    copies it up with cudaMemcpy HostToDevice), so a host-side loop over
    //    st.page_table would fault. kv_stream_eject also frees the slots those
    //    blocks held, so the residency clock can reuse them for the tail.
    const long long retired = strata::kernels::kv_stream_eject(st.map, (long long) cut_pages, stream);
    if (retired < 0) {
        rep.note = "could not run the device eject (no counter allocation)";
        return rep;
    }
    rep.pages_ejected = retired;
    rep.cells_ejected = (uint64_t) (cut_pages * s.page_size);

    // 2. Release the host cells. Each pool pointer is indexed by the same cell
    //    coordinate, so one pass covers all of them.
    const int fmt = qsa_kv_format(st);
    const uint64_t cell_bytes = strata::kernels::kv_block_bytes(s, fmt);
    const uint64_t bytes = cell_bytes * (uint64_t) (cut_pages * s.page_size);

    void* pools[] = {
        (void*) st.host.k_pool, (void*) st.host.v_pool,
        (void*) st.host.k_q,    (void*) st.host.v_q,
        (void*) st.host.k_scale, (void*) st.host.v_scale,
        (void*) st.host.k_q4,   (void*) st.host.v_q4,
    };
    for (void* p : pools) {
        if (p == nullptr) continue;
        const uint64_t got = release_range(p, bytes);
        rep.bytes_released += got;
        if (got == 0) rep.bytes_locked += bytes;
    }
    rep.pages_released = cell_bytes > 0 ? rep.bytes_released / cell_bytes : 0;

    // 3. Give the allocator's freed bytes a chance to leave the process.
    trim_heap();

    rep.ok = true;
    if (rep.bytes_locked > 0 && rep.bytes_released == 0) {
        rep.note = "cells marked non-resident; host pools stayed resident (pinned)";
    } else {
        rep.note = "cells marked non-resident and released where not pinned";
    }
    std::fprintf(stderr,
                 "strata: kv eject: cut=%lld cells, %lld pages retired, %llu MiB released, "
                 "%llu MiB stayed pinned\n",
                 (long long) cut_cells, (long long) rep.pages_ejected,
                 (unsigned long long) (rep.bytes_released / (1u << 20)),
                 (unsigned long long) (rep.bytes_locked / (1u << 20)));
    return rep;
}

EjectReport kv_eject_all_parked(std::string& note) {
    EjectReport rep;
    note = "no parked conversations to release";
    rep.ok = true;
    return rep;
}

}  // namespace strata::core