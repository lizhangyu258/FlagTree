// RUN: triton-opt %s --tle-convert-arg-to-memdesc | FileCheck %s

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#smem = #ttg.shared_memory

module {
  // CHECK-LABEL: tt.func @explicit_effects
  // CHECK-NOT: nvvm.barrier0
  // CHECK: "tle.dsl_region"
  tt.func @explicit_effects(%slot: !ttg.memdesc<16xf32, #shared1, #smem, mutable>) {
    "tle.dsl_region"(%slot) ({
    ^bb0(%arg0: !ttg.memdesc<16xf32, #shared1, #smem, mutable>):
      tle.yield
    }) {arg_dialect = "llvm", output_operand_indices = array<i32>, region_dialect = "cuda", tle_raw.arg_effects = array<i32: 1>} : (!ttg.memdesc<16xf32, #shared1, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK-LABEL: tt.func @legacy_effects
  // CHECK: nvvm.barrier0
  // CHECK-NEXT: "tle.dsl_region"
  tt.func @legacy_effects(%slot: !ttg.memdesc<16xf32, #shared1, #smem, mutable>) {
    "tle.dsl_region"(%slot) ({
    ^bb0(%arg0: !ttg.memdesc<16xf32, #shared1, #smem, mutable>):
      tle.yield
    }) {arg_dialect = "llvm", output_operand_indices = array<i32>, region_dialect = "cuda"} : (!ttg.memdesc<16xf32, #shared1, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK-LABEL: tt.func @tensor_staging
  // CHECK: %[[ALLOC:.*]] = ttg.local_alloc
  // CHECK: ttg.local_store {{.*}}, %[[ALLOC]]
  // CHECK: nvvm.barrier0
  // CHECK-NEXT: "tle.dsl_region"
  tt.func @tensor_staging(%value: tensor<16xf32, #blocked>) {
    "tle.dsl_region"(%value) ({
    ^bb0(%arg0: tensor<16xf32, #blocked>):
      tle.yield
    }) {arg_dialect = "llvm", output_operand_indices = array<i32>, region_dialect = "cuda", tle_raw.arg_effects = array<i32: 1>} : (tensor<16xf32, #blocked>) -> ()
    tt.return
  }
}
