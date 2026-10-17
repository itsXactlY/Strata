#pragma once
// CUDA virtual memory (the elastic K/V, --kv-grow): an address range reserved once, whose physical memory is mapped and
// unmapped in chunks of the device's granularity (2 MiB).  The K/V pools grow with the context by taking chunks the
// expert cache gives up, and neither moves: every captured graph and stored pointer stays valid.  The driver calls
// are reached through the runtime's entry-point query, so nothing links the driver library.  Not under HIP.

#include <cstdint>
#include <vector>

namespace strata::core {

/// A physical chunk: the driver's allocation handle (CUmemGenericAllocationHandle); 0 = none.
using VmmChunk = unsigned long long;

/// The current device supports it and the entry points resolved.
bool vmm_available();
/// The chunk size (0 when not available).
uint64_t vmm_granularity();
/// A new physical chunk on the current device (0: out of memory), and its release.
VmmChunk vmm_chunk_new();
void vmm_chunk_free(VmmChunk h);
/// Host-located chunks (--kv-shared): pinned RAM the device reads and writes through the range, the CPU too.
bool vmm_host_available();
uint64_t vmm_host_granularity();
VmmChunk vmm_host_chunk_new();

class VmmRange {
public:
    VmmRange() = default;
    ~VmmRange() { release(); }
    VmmRange(const VmmRange&) = delete;
    VmmRange& operator=(const VmmRange&) = delete;

    /// Reserves `bytes` (rounded up to whole chunks) of addresses, nothing mapped.  `host`: its chunks are host-located
    /// (new ones from vmm_host_chunk_new) and the CPU may access them as well.
    bool reserve(uint64_t bytes, bool host = false);
    /// Unmaps and frees every chunk, then the addresses.
    void release();
    uint8_t* base() const { return (uint8_t*) (uintptr_t) base_; }
    int64_t chunks() const { return (int64_t) h_.size(); }
    bool mapped(int64_t i) const { return i >= 0 && i < chunks() && h_[(std::size_t) i] != 0; }
    int64_t mapped_count() const;
    /// Maps chunks [lo, hi) that are not mapped yet, each to `take()` or, when that returns 0, to a new chunk.
    /// Readable and writable when it returns; false: out of memory (what was mapped stays mapped).
    template <class Take> bool map_range(int64_t lo, int64_t hi, Take take);
    /// Unmaps chunk i and hands back its physical chunk (0 when it was not mapped).
    VmmChunk unmap(int64_t i);
    /// Unmaps every chunk and hands back the physical ones by index (0: was not mapped).
    std::vector<VmmChunk> take_all();
    /// Maps hs[i] at chunk i for every non-zero entry (the range holds none mapped there); false: a map failed,
    /// and the chunks it could not place are freed.  With take_all() this moves physical memory between ranges.
    bool adopt(const std::vector<VmmChunk>& hs);

private:
    bool map_one(int64_t i, VmmChunk h);
    bool set_access(int64_t lo, int64_t hi);
    /// Readable and writable [lo, hi) (all newly mapped); when that fails they are unmapped and freed again, so no
    /// chunk counts as mapped without access.
    bool commit_run(int64_t lo, int64_t hi);
    unsigned long long base_ = 0;
    uint64_t gran_ = 0;   // this range's chunk size (the device's or the host's granularity)
    bool host_ = false;
    std::vector<VmmChunk> h_;   // per chunk: its physical chunk, 0 = unmapped
};

template <class Take> bool VmmRange::map_range(int64_t lo, int64_t hi, Take take) {
    lo = lo < 0 ? 0 : lo;
    hi = hi > chunks() ? chunks() : hi;
    int64_t run = -1;   // access is set once per run of newly mapped chunks (it costs as much as the map)
    for (int64_t i = lo; i < hi; ++i) {
        if (mapped(i)) {
            if (run >= 0 && !commit_run(run, i)) return false;
            run = -1;
            continue;
        }
        VmmChunk h = take();
        if (h == 0) h = host_ ? vmm_host_chunk_new() : vmm_chunk_new();
        if (h == 0 || !map_one(i, h)) {
            if (h != 0) vmm_chunk_free(h);
            if (run >= 0) commit_run(run, i);
            return false;
        }
        if (run < 0) run = i;
    }
    return run < 0 || commit_run(run, hi);
}

}  // namespace strata::core
