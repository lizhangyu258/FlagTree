// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// RUN: triton-opt %s --triton-tle-lower-pipe-to-nvws | FileCheck %s

#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @raw_pipe_round_trip
  // CHECK: %[[TOKEN:.*]] = nvws.create_token
  // CHECK-SAME: full_count = 128 : i32
  // CHECK: nvws.producer_acquire %[[TOKEN]]
  // CHECK: %[[RAW_ALIAS:.*]] = "tle.dsl_region"
  // CHECK-SAME: tle_raw.arg_effects = array<i32: 3>
  // CHECK: nvws.producer_commit %[[TOKEN]]
  // CHECK-SAME: commitKind = 3 : i32
  // CHECK: nvws.consumer_wait %[[TOKEN]]
  // CHECK: "tle.dsl_region"(%[[RAW_ALIAS]])
  // CHECK-SAME: tle_raw.arg_effects = array<i32: 1>
  // CHECK: nvws.consumer_release %[[TOKEN]]
  tt.func @raw_pipe_round_trip(%data: !ttg.memdesc<1x16xf32, #shared2, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %data {capacity = 1 : i32, pipe_name = "raw", field_names = ["x"], scope = "cta"} : !ttg.memdesc<1x16xf32, #shared2, #smem, mutable>
    tle.pipe.writer_acquire %data[%c0, %false] {capacity = 1 : i32, pipe_name = "raw", field_names = ["x"], scope = "cta"} : !ttg.memdesc<1x16xf32, #shared2, #smem, mutable>
    %slot = ttg.memdesc_index %data[%c0] : !ttg.memdesc<1x16xf32, #shared2, #smem, mutable> -> !ttg.memdesc<16xf32, #shared1, #smem, mutable>
    %alias = "tle.dsl_region"(%slot) ({
    ^bb0(%arg0: !ttg.memdesc<16xf32, #shared1, #smem, mutable>):
      tle.yield %arg0 : !ttg.memdesc<16xf32, #shared1, #smem, mutable>
    }) {arg_dialect = "llvm", output_operand_indices = array<i32: 0>,
        region_dialect = "cuda", tle_raw.arg_effects = array<i32: 3>} :
        (!ttg.memdesc<16xf32, #shared1, #smem, mutable>) ->
        !ttg.memdesc<16xf32, #shared1, #smem, mutable>
    tle.pipe.writer_commit %data[%c0] {capacity = 1 : i32, pipe_name = "raw", field_names = ["x"], scope = "cta"} : !ttg.memdesc<1x16xf32, #shared2, #smem, mutable>
    %closed = tle.pipe.reader_wait %data[%c0, %false] {capacity = 1 : i32, pipe_name = "raw", field_names = ["x"], scope = "cta"} : !ttg.memdesc<1x16xf32, #shared2, #smem, mutable>
    "tle.dsl_region"(%alias) ({
    ^bb0(%arg0: !ttg.memdesc<16xf32, #shared1, #smem, mutable>):
      tle.yield
    }) {arg_dialect = "llvm", output_operand_indices = array<i32>,
        region_dialect = "cuda", tle_raw.arg_effects = array<i32: 1>} :
        (!ttg.memdesc<16xf32, #shared1, #smem, mutable>) -> ()
    tle.pipe.reader_release %data[%c0] {capacity = 1 : i32, pipe_name = "raw", field_names = ["x"], scope = "cta"} : !ttg.memdesc<1x16xf32, #shared2, #smem, mutable>
    tt.return
  }
}
