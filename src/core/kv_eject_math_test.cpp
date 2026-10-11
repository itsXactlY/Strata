// Host-only test of kv_eject_math.hpp: g++ -std=c++20 -Iinclude src/core/kv_eject_math_test.cpp && ./a.out
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#include "strata/core/kv_eject_math.hpp"

using namespace strata::core;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // int8, 4 KV heads, 256-cell pages, head_dim 128, scale group 64
    const uint64_t rows = 4 * 256, hd = 128;
    const uint64_t block = rows * hd * 2 + rows * (hd / 64) * 2 * 2;      // as kv_block_bytes(int8)
    PoolPageBytes b = pool_page_bytes(1, block, rows, hd);
    CHECK(b.codes == rows * hd);
    CHECK(b.scales == rows * (hd / 64) * 2);
    CHECK(2 * b.codes + 2 * b.scales == block);                              // the four arrays tile one block

    const uint64_t fp16_block = rows * hd * 2 * 2;
    b = pool_page_bytes(0, fp16_block, rows, hd);
    CHECK(b.codes == rows * hd * 2 && b.scales == 0 && 2 * b.codes == fp16_block);

    const uint64_t q4_block = rows * 72 * 2;                                  // 72 B per head row (q4_0)
    b = pool_page_bytes(2, q4_block, rows, hd);
    CHECK(b.codes == rows * 72 && b.scales == 0);

    // the old formula: block_bytes * cut_cells for a 130-page cut, per array
    const uint64_t cut_pages = 130, cut_cells = cut_pages * 256;
    const uint64_t array_bytes_old = block * cut_cells;
    const uint64_t array_bytes_new = cut_pages * pool_page_bytes(1, block, rows, hd).codes;
    CHECK(array_bytes_old == 528 * array_bytes_new);                          // (block / codes) * page_size, int8 here

    // inward alignment
    const uintptr_t P = 4096;
    HostRange r = inward_page_range(4096, 8192, P);
    CHECK(r.begin == 4096 && r.end == 12288);                                 // already aligned: whole range
    r = inward_page_range(4096 + 100, 8192, P);                               // starts mid-page, ends mid-page
    CHECK(r.begin == 8192 && r.end == 12288);                                 // both partial pages are kept
    r = inward_page_range(4096 + 100, 3000, P);
    CHECK(r.size() == 0);                                                     // inside one OS page: release nothing
    r = inward_page_range(0, 0, P);
    CHECK(r.size() == 0);
    // a released range never reaches a byte outside [base, base + bytes)
    for (uintptr_t base : {0x1000ul, 0x1001ul, 0x1ffful, 0x7f12345678ul}) {
        for (uint64_t bytes : {1ul, 4095ul, 4096ul, 8191ul, 100000ul}) {
            HostRange q = inward_page_range(base, bytes, P);
            if (q.size()) CHECK(q.begin >= base && q.end <= base + bytes && q.begin % P == 0 && q.end % P == 0);
        }
    }
    if (fails == 0) std::puts("kv_eject_math: ok");
    return fails ? 1 : 0;
}
