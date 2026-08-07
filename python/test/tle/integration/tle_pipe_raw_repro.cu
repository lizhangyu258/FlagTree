#include <stdint.h>

using shared_float_ptr = __attribute__((address_space(3))) float *;
using global_float_ptr = __attribute__((address_space(1))) float *;

static __device__ __attribute__((always_inline)) int64_t
memref_index(int64_t offset, int64_t row, int64_t col, int64_t stride0,
             int64_t stride1) {
  return offset + row * stride0 + col * stride1;
}

static __device__ __attribute__((always_inline)) float
expected_value(int iteration, int field, int64_t row, int64_t col) {
  return static_cast<float>(iteration * 100000 + field * 10000 + row * 100 +
                            col);
}

// Deliberately uses CTA-global threadIdx.x/blockDim.x inside a warp-specialized
// worker. The memref offset and strides are honored so failures are not caused
// by accidentally addressing stage zero. The second field is nevertheless
// treated as an ordinary strided array even though its TLE allocation uses an
// NV-MMA shared-memory layout.
extern "C" __device__ __attribute__((always_inline)) void raw_pipe_producer(
    shared_float_ptr plain_allocated, shared_float_ptr plain_aligned,
    int64_t plain_offset, int64_t plain_size0, int64_t plain_size1,
    int64_t plain_stride0, int64_t plain_stride1,
    shared_float_ptr swizzled_allocated, shared_float_ptr swizzled_aligned,
    int64_t swizzled_offset, int64_t swizzled_size0, int64_t swizzled_size1,
    int64_t swizzled_stride0, int64_t swizzled_stride1, int iteration) {
  (void)plain_allocated;
  (void)swizzled_allocated;

  const int64_t elements = plain_size0 * plain_size1;
  for (int64_t linear = threadIdx.x; linear < elements;
       linear += blockDim.x) {
    const int64_t row = linear / plain_size1;
    const int64_t col = linear % plain_size1;
    plain_aligned[memref_index(plain_offset, row, col, plain_stride0,
                               plain_stride1)] =
        expected_value(iteration, 0, row, col);

    if (row < swizzled_size0 && col < swizzled_size1) {
      swizzled_aligned[memref_index(swizzled_offset, row, col,
                                   swizzled_stride0, swizzled_stride1)] =
          expected_value(iteration, 1, row, col);
    }
  }
}

// Synchronously reads pipe fields from a worker partition and writes global
// output. This avoids TMA/cp.async so the repro isolates partition thread
// semantics and shared-layout interpretation. As above, the memref offset is
// used exactly as passed by TLE.
extern "C" __device__ __attribute__((always_inline)) void raw_pipe_consumer(
    shared_float_ptr plain_allocated, shared_float_ptr plain_aligned,
    int64_t plain_offset, int64_t plain_size0, int64_t plain_size1,
    int64_t plain_stride0, int64_t plain_stride1,
    shared_float_ptr swizzled_allocated, shared_float_ptr swizzled_aligned,
    int64_t swizzled_offset, int64_t swizzled_size0, int64_t swizzled_size1,
    int64_t swizzled_stride0, int64_t swizzled_stride1,
    global_float_ptr output, int iteration) {
  (void)plain_allocated;
  (void)swizzled_allocated;

  const int64_t elements = plain_size0 * plain_size1;
  const int64_t iteration_base = static_cast<int64_t>(iteration) * 2 * elements;
  for (int64_t linear = threadIdx.x; linear < elements;
       linear += blockDim.x) {
    const int64_t row = linear / plain_size1;
    const int64_t col = linear % plain_size1;
    output[iteration_base + linear] =
        plain_aligned[memref_index(plain_offset, row, col, plain_stride0,
                                   plain_stride1)];

    if (row < swizzled_size0 && col < swizzled_size1) {
      output[iteration_base + elements + linear] =
          swizzled_aligned[memref_index(swizzled_offset, row, col,
                                       swizzled_stride0, swizzled_stride1)];
    }
  }
}
