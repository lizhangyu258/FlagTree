// RUN: triton-opt %s -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=90 ptx-version=80}, tritongpu-global-scratch-memory-allocation, tle-dslregion-inline, convert-triton-gpu-to-llvm{compute-capability=90 ptx-version=80}, canonicalize, cse, convert-nv-gpu-to-llvm, convert-warp-specialize-to-llvm, canonicalize, cse, symbol-dce, convert-nvvm-to-llvm)' | FileCheck %s

#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  llvm.func @consume_memdesc(!llvm.ptr<3>, !llvm.ptr<3>, i64, i64, i64)

  tt.func public @extract_indexed_memdesc() {
    %c0 = arith.constant 0 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x256xf32, #shared2, #smem, mutable>
    %slot = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x256xf32, #shared2, #smem, mutable> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
    "tle.dsl_region"(%slot) ({
    ^bb0(%arg0: !ttg.memdesc<256xf32, #shared1, #smem, mutable>):
      %allocated = tle.extract_allocated_ptr %arg0 : !ttg.memdesc<256xf32, #shared1, #smem, mutable> to !llvm.ptr<3>
      %aligned = tle.extract_aligned_ptr %arg0 : !ttg.memdesc<256xf32, #shared1, #smem, mutable> to !llvm.ptr<3>
      %offset = tle.extract_offset %arg0 : !ttg.memdesc<256xf32, #shared1, #smem, mutable> to i64
      %size = tle.extract_sizes %arg0 : !ttg.memdesc<256xf32, #shared1, #smem, mutable> to i64
      %stride = tle.extract_strides %arg0 : !ttg.memdesc<256xf32, #shared1, #smem, mutable> to i64
      llvm.call @consume_memdesc(%allocated, %aligned, %offset, %size, %stride) : (!llvm.ptr<3>, !llvm.ptr<3>, i64, i64, i64) -> ()
      tle.yield
    }) {arg_dialect = "llvm", output_operand_indices = array<i32>, region_dialect = "cuda", tle_raw.arg_effects = array<i32: 1>} : (!ttg.memdesc<256xf32, #shared1, #smem, mutable>) -> ()
    tt.return
  }
}

// CHECK-LABEL: llvm.func @extract_indexed_memdesc
// CHECK-NOT: tle.dsl_region
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: tle.extract_
// CHECK: llvm.call @consume_memdesc
// CHECK-NOT: unrealized_conversion_cast
