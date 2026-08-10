#include <stdint.h>

struct MemDesc1D {
  __attribute__((address_space(3))) float *allocated;
  __attribute__((address_space(3))) float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};

extern "C" __device__ MemDesc1D raw_pipe_producer(
    __attribute__((address_space(3))) float *allocated,
    __attribute__((address_space(3))) float *aligned, int64_t offset,
    int64_t size, int64_t stride,
    __attribute__((address_space(1))) const float *input) {
  // The integration test launches four warps for a 128-element tile, so every
  // CTA thread owns exactly one element. Keeping this call branch-free also
  // exercises the single-block eager dsl_region lowering boundary.
  int32_t index = threadIdx.x;
  aligned[offset + static_cast<int64_t>(index) * stride] = input[index];
  return {allocated, aligned, offset, {size}, {stride}};
}

extern "C" __device__ MemDesc1D raw_pipe_consumer(
    __attribute__((address_space(3))) float *allocated,
    __attribute__((address_space(3))) float *aligned, int64_t offset,
    int64_t size, int64_t stride,
    __attribute__((address_space(1))) float *output) {
  int32_t index = threadIdx.x;
  output[index] = aligned[offset + static_cast<int64_t>(index) * stride];
  return {allocated, aligned, offset, {size}, {stride}};
}
