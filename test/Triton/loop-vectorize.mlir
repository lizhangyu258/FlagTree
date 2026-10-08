// RUN: triton-opt %s -triton-loop-vectorize -canonicalize | FileCheck %s
// RUN: triton-opt %s -triton-loop-vectorize -triton-loop-unroll -canonicalize | FileCheck %s

module {
  // CHECK-LABEL: @vectorize_contract
  // CHECK-NOT: scf.for
  // CHECK-NOT: tt.loop_unroll_factor
  // CHECK-NOT: tt.reorder
  // CHECK-COUNT-1: tt.load
  // CHECK-DAG: tensor<4x64xf32>
  // CHECK-DAG: "tt.reduce"
  // CHECK-DAG: tt.store
  tt.func public @vectorize_contract(
      %x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %scale: !tt.ptr<f32>,
      %group_size: i32, %first_group: i32) {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 4 : i32
    %step = arith.constant 1 : i32
    %cols = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %group_size_tensor = tt.splat %group_size : i32 -> tensor<64xi32>
    %mask = arith.cmpi slt, %cols, %group_size_tensor : tensor<64xi32>
    %other = arith.constant dense<0.0> : tensor<64xf32>
    scf.for %i = %lb to %ub step %step : i32 {
      %group = arith.addi %first_group, %i : i32
      %group_offset = arith.muli %group, %group_size : i32
      %xp0 = tt.addptr %x, %group_offset : !tt.ptr<f32>, i32
      %xp1 = tt.splat %xp0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %xp = tt.addptr %xp1, %cols : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %value = tt.load %xp, %mask, %other : tensor<64x!tt.ptr<f32>>
      %max = "tt.reduce"(%value) <{axis = 0 : i32}> ({
      ^bb0(%lhs: f32, %rhs: f32):
        %result = arith.maxnumf %lhs, %rhs : f32
        tt.reduce.return %result : f32
      }) : (tensor<64xf32>) -> f32
      %yp0 = tt.addptr %y, %group_offset : !tt.ptr<f32>, i32
      %yp1 = tt.splat %yp0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %yp = tt.addptr %yp1, %cols : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %yp, %value, %mask : tensor<64x!tt.ptr<f32>>
      %sp = tt.addptr %scale, %group : !tt.ptr<f32>, i32
      tt.store %sp, %max : !tt.ptr<f32>
    } {tt.loop_vectorize_factor = 4 : i32}
    tt.return
  }

  // A runtime upper bound is strip-mined into a vector main loop and a scalar
  // epilogue.  The vector hint is consumed and must not reach LoopUnroll.
  // CHECK-LABEL: @vectorize_runtime_upper
  // CHECK-NOT: tt.loop_vectorize_factor
  // CHECK-NOT: tt.loop_unroll_factor
  // CHECK: scf.for
  // CHECK: tt.load {{.*}} : tensor<4x64x!tt.ptr<f32>>
  // CHECK: tt.store {{.*}} : tensor<4x64x!tt.ptr<f32>>
  // CHECK: scf.for
  // CHECK: tt.load {{.*}} : tensor<64x!tt.ptr<f32>>
  // CHECK: tt.store {{.*}} : tensor<64x!tt.ptr<f32>>
  tt.func public @vectorize_runtime_upper(
      %x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %ub: i32,
      %group_size: i32) {
    %lb = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %cols = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    scf.for %i = %lb to %ub step %step : i32 {
      %offset = arith.muli %i, %group_size : i32
      %xp0 = tt.addptr %x, %offset : !tt.ptr<f32>, i32
      %xp1 = tt.splat %xp0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %xp = tt.addptr %xp1, %cols : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %value = tt.load %xp : tensor<64x!tt.ptr<f32>>
      %yp0 = tt.addptr %y, %offset : !tt.ptr<f32>, i32
      %yp1 = tt.splat %yp0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %yp = tt.addptr %yp1, %cols : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %yp, %value : tensor<64x!tt.ptr<f32>>
    } {tt.loop_vectorize_factor = 4 : i32}
    tt.return
  }
}
