#include <cuda_fp16.h>
#include <stdint.h>

using global_half_ptr = __attribute__((address_space(1))) const half *;
using shared_half_ptr = __attribute__((address_space(3))) half *;

static __attribute__((device, always_inline)) void
copy_global_to_shared_16_async(shared_half_ptr dst, global_half_ptr src,
                               uint32_t valid_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :
               : "r"(dst), "l"(src), "r"(valid_bytes)
               : "memory");
}

// Fill the two memrefs supplied by tle_raw.call and return their descriptors.
// Both shared descriptors must have stride1 == 1, as selected by shared_orders
// in the caller. This standalone loader completes all copies before returning.
__attribute__((device)) auto
LoadTiles(shared_half_ptr a_allocated, shared_half_ptr a_aligned,
          const int64_t a_offset, const int64_t a_size0, const int64_t a_size1,
          const int64_t a_stride0, const int64_t a_stride1,
          shared_half_ptr b_allocated, shared_half_ptr b_aligned,
          const int64_t b_offset, const int64_t b_size0, const int64_t b_size1,
          const int64_t b_stride0, const int64_t b_stride1, global_half_ptr a,
          global_half_ptr b, const int M, const int N, const int K,
          const int stride_am, const int stride_ak, const int stride_bk,
          const int stride_bn, const int pid_m, const int pid_n, const int k) {
  uint32_t tid;
  uint32_t num_threads;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
  asm volatile("mov.u32 %0, %%ntid.x;" : "=r"(num_threads));
  const int64_t k_begin = static_cast<int64_t>(k) * a_size1;

  constexpr int64_t async_elements = 8;
  const int64_t b_tile_col_begin = static_cast<int64_t>(pid_n) * b_size1;
  const int64_t a_vectors = a_size0 * a_size1 / async_elements;
  for (int64_t vector = tid; vector < a_vectors; vector += num_threads) {
    const int64_t linear = vector * async_elements;
    const int64_t row = linear / a_size1;
    const int64_t col = linear % a_size1;
    const int64_t global_row =
        (static_cast<int64_t>(pid_m) * a_size0 + row) % M;
    const int64_t global_k = k_begin + col;
    const int64_t remaining = static_cast<int64_t>(K) - global_k;
    const uint32_t valid_elements =
        remaining <= 0
            ? 0
            : static_cast<uint32_t>(
                  remaining < async_elements ? remaining : async_elements);
    global_half_ptr src =
        valid_elements == 0 ? a
                            : a + global_row * stride_am + global_k * stride_ak;
    shared_half_ptr dst =
        a_aligned + a_offset + row * a_stride0 + col * a_stride1;
    copy_global_to_shared_16_async(dst, src, valid_elements * sizeof(half));
  }

  const int64_t b_vectors = b_size0 * b_size1 / async_elements;
  for (int64_t vector = tid; vector < b_vectors; vector += num_threads) {
    const int64_t linear = vector * async_elements;
    const int64_t row = linear / b_size1;
    const int64_t col = linear % b_size1;
    const int64_t global_k = k_begin + row;
    const int64_t global_col = b_tile_col_begin + col;
    const int64_t remaining = static_cast<int64_t>(N) - global_col;
    const uint32_t valid_elements =
        global_k >= K || remaining <= 0
            ? 0
            : static_cast<uint32_t>(
                  remaining < async_elements ? remaining : async_elements);
    const uint32_t valid_bytes = valid_elements * sizeof(half);
    global_half_ptr src =
        valid_bytes == 0 ? b
                         : b + global_k * stride_bk + global_col * stride_bn;
    shared_half_ptr dst =
        b_aligned + b_offset + row * b_stride0 + col * b_stride1;
    copy_global_to_shared_16_async(dst, src, valid_bytes);
  }

  asm volatile("cp.async.commit_group;\n"
               "cp.async.wait_group 0;\n"
               "bar.sync 0;" ::
                   : "memory");

  struct MemRef2D {
    shared_half_ptr allocated;
    shared_half_ptr aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
  };
  struct {
    MemRef2D a;
    MemRef2D b;
  } result{
      {a_allocated,
       a_aligned,
       a_offset,
       {a_size0, a_size1},
       {a_stride0, a_stride1}},
      {b_allocated,
       b_aligned,
       b_offset,
       {b_size0, b_size1},
       {b_stride0, b_stride1}},
  };
  return result;
}
