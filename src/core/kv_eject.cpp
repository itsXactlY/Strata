// src/core/kv_eject.cpp - see include/strata/core/kv_eject.hpp for the contract.
//
// The host pools are one cudaHostAlloc region carved into SEPARATE k, v (and scale) arrays, each indexed by
// page. A cut of N pages is therefore the first N pages of EACH array: N * that array's bytes per page
// (kv_eject_math.hpp), shrunk inward to whole OS pages so no byte of a kept cell is ever released.

#include "strata/core/kv_eject.hpp"
#include "strata/core/kv_eject_math.hpp"

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
/// Release [begin, end), already shrunk inward to whole OS pages (see inward_page_range). Returns the bytes
/// actually released: madvise on a page-locked region fails with EINVAL, and that is reported as "stayed
/// resident" rather than hidden.
uint64_t release_range(uintptr_t begin, uintptr_t end) {
    if (end <= begin) return 0;
    if (madvise((void*) begin, (size_t) (end - begin), MADV_DONTNEED) != 0) return 0;
    return (uint64_t) (end - begin);
}
uintptr_t os_page_size() { return (uintptr_t) sysconf(_SC_PAGESIZE); }
#else
uint64_t release_range(uintptr_t, uintptr_t) { return 0; }
uintptr_t os_page_size() { return 4096; }
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
    const uint64_t block_bytes = strata::kernels::kv_block_bytes(s, fmt);
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    const PoolPageBytes pb = pool_page_bytes(fmt, block_bytes, rows, (uint64_t) s.head_dim);

    // Every host array is its own contiguous run indexed by page, so the range to release is the first
    // `cut_pages` pages of THAT array: cut_pages * (the array's bytes per page). Sizing them all with
    // block_bytes * cut_cells covered the neighbouring arrays and the end of the region.
    struct Pool { void* p; uint64_t page_bytes; };
    const Pool pools[] = {
        {(void*) st.host.k_pool, pb.codes}, {(void*) st.host.v_pool, pb.codes},
        {(void*) st.host.k_q, pb.codes},    {(void*) st.host.v_q, pb.codes},
        {(void*) st.host.k_scale, pb.scales}, {(void*) st.host.v_scale, pb.scales},
        {(void*) st.host.k_q4, pb.codes},   {(void*) st.host.v_q4, pb.codes},
    };
    for (const Pool& pool : pools) {
        if (pool.p == nullptr || pool.page_bytes == 0) continue;
        const HostRange r = inward_page_range((uintptr_t) pool.p, (uint64_t) cut_pages * pool.page_bytes,
                                              os_page_size());
        if (r.size() == 0) continue;                  // less than one whole OS page: nothing may be released
        const uint64_t got = release_range(r.begin, r.end);
        rep.bytes_released += got;
        if (got == 0) rep.bytes_locked += r.size();
    }
    rep.pages_released = block_bytes > 0 ? rep.bytes_released / block_bytes : 0;

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