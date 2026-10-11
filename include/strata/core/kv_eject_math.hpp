// include/strata/core/kv_eject_math.hpp - the byte arithmetic of kv_eject, free of CUDA so it can be tested.
//
// kv_eject used to size every pool as kv_block_bytes * cut_cells. kv_block_bytes is the bytes of ONE PAGE
// (n_head_kv * page_size rows of K AND V and their scales) and the host region is carved into separate K, V and
// scale arrays, so that range was page_size times too long and started at every pool's own base: it covered the
// neighbouring pools and ran past the end of the region. It also rounded the range OUTWARD to the OS page, which
// discards up to 4 KiB of the first kept row. The two helpers below are the corrected sizes.
#pragma once

#include <cstdint>

namespace strata::core {

/// Bytes ONE page (one KV block of page_size cells, one layer) occupies in each host array.
struct PoolPageBytes {
    uint64_t codes = 0;    ///< k/v data: fp16 values, int8 codes, or q4_0 blocks, per array
    uint64_t scales = 0;   ///< k/v scale array (int8 only), per array
};

/// `fmt` is KvFormat (0 fp16, 1 int8, 2 q4_0); `block_bytes` is kv_block_bytes(s, fmt); `rows` is
/// n_head_kv * page_size. K and V are the same size, so each array holds half of block_bytes' data.
inline PoolPageBytes pool_page_bytes(int fmt, uint64_t block_bytes, uint64_t rows, uint64_t head_dim) {
    PoolPageBytes b;
    if (fmt == 1) {                                  // int8: codes (1 B per value) + fp16 scales
        b.codes = rows * head_dim;
        b.scales = block_bytes / 2 - b.codes;
    } else {                                         // fp16 and q4_0: K and V split the block evenly
        b.codes = block_bytes / 2;
    }
    return b;
}

struct HostRange {
    uintptr_t begin = 0;
    uintptr_t end = 0;
    uint64_t size() const { return end > begin ? (uint64_t) (end - begin) : 0; }
};

/// [base, base + bytes) shrunk INWARD to whole OS pages: only a page that lies entirely inside the range may be
/// returned to the allocator, because the bytes next to it belong to cells that stay.
inline HostRange inward_page_range(uintptr_t base, uint64_t bytes, uintptr_t page) {
    HostRange r;
    if (bytes == 0 || page == 0) return r;
    const uintptr_t lo = (base + page - 1) & ~(page - 1);
    const uintptr_t hi = (base + (uintptr_t) bytes) & ~(page - 1);
    if (hi > lo) { r.begin = lo; r.end = hi; }
    return r;
}

}  // namespace strata::core
