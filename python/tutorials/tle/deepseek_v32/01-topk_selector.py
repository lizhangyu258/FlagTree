"""
DeepSeek V3-2 Top-K Selector with Triton and TLE (TLE Tutorial)
==============================================================

This tutorial adapts the TileLang DeepSeek V3-2 top-k selector example and
implements two kernels:
- A Triton version rewritten with the radix-select flow used in `03-topk.py`.
- A TLE version that keeps the shared-memory DeepSeek-style selector
  (`tle.gpu.alloc` + `tle.gpu.local_ptr`).

If TileLang is installed, the script will also run the original TileLang kernel
and compare correctness and performance.

Notes
-----
- Input dtype is assumed to be float32 for the 32-bit radix refinement.
- `SMEM_INPUT_SIZE` bounds the number of candidates carried into stage-2.
  If the threshold bucket exceeds this size, results are approximate.
"""

# %%
# Setup
# -----

import argparse
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

try:
    import tilelang
    import tilelang.language as T

    _HAVE_TILELANG = True
except Exception:  # pragma: no cover - optional dependency
    tilelang = None
    T = None
    _HAVE_TILELANG = False

DEVICE = triton.runtime.driver.active.get_active_torch_device()
RADIX_BITS = 8
RADIX = 1 << RADIX_BITS
TLE_TOPK_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=8, num_stages=1),
    triton.Config({}, num_warps=16, num_stages=1),
    triton.Config({}, num_warps=32, num_stages=1),
]

# %%
# Key conversions
# ---------------


@triton.jit
def _convert_to_uint16(x):
    hval = x.to(tl.float16)
    bits = hval.to(tl.uint16, bitcast=True)
    sign_mask = tl.full(hval.shape, 0x8000, tl.uint16)
    bits = tl.where(x < 0, ~bits, bits | sign_mask)
    return bits >> 8


@triton.jit
def _convert_to_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    bits = tl.where(x < 0, ~bits, bits | sign_mask)
    return bits


# %%
# Triton kernel (radix-select, based on 03-topk)
# ------------------------------


@triton.jit
def triton_topk_selector_kernel(
    x_ptr,
    out_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    num_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    stride_hist,
    stride_num,
    seq_len,
    RADIX_BITS: tl.constexpr,
    ASSUME_ALIGNED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = tl.load(starts_ptr + pid).to(tl.int32)
    row_end = tl.load(ends_ptr + pid).to(tl.int32)

    row_ptr = x_ptr + pid * stride_xm
    out_row = out_ptr + pid * stride_outm
    hist_row = hist_ptr + pid * stride_hist
    num_row = num_ptr + pid * stride_num

    if ASSUME_ALIGNED:
        tl.assume(row_start == 0)
        tl.assume(row_end == seq_len)
        tl.assume(stride_xn == 1)
        tl.assume(stride_outn == 1)
        seq_len = tl.multiple_of(seq_len, BLOCK_SIZE)

    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
    RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
    hist_idx = tl.arange(0, RADIX_SIZE)

    desired = tl.full((), 0, dtype=tl.uint32)
    desired_mask = tl.full((), 0, dtype=tl.uint32)
    k_to_find = tl.full((), TOPK, dtype=tl.int32)
    n_tiles = tl.cdiv(seq_len, BLOCK_SIZE)

    # MSD radix-select on 32-bit float keys.
    for digit_pos in tl.static_range(32 - RADIX_BITS, -1, -RADIX_BITS):
        tl.store(hist_row + hist_idx, 0)
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
            x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
            x_key = _convert_to_uint32(x)
            matches = (x_key & desired_mask) == desired
            digit = ((x_key >> digit_pos) & RADIX_MASK).to(tl.int32)
            tl.atomic_add(hist_row + digit, ones, mask=in_range & matches)

        counts = tl.load(hist_row + hist_idx)
        cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
        tl.store(hist_row + hist_idx, cumsum_desc)

        cond = cumsum_desc >= k_to_find
        selected = tl.max(tl.where(cond, hist_idx, 0), axis=0).to(tl.int32)
        counts_gt = tl.max(tl.where(hist_idx == (selected + 1), cumsum_desc, 0), axis=0)

        selected_u = selected.to(tl.uint32)
        desired = desired | (selected_u << digit_pos)
        desired_mask = desired_mask | (tl.full((), RADIX_MASK, dtype=tl.uint32) << digit_pos)
        k_to_find = k_to_find - counts_gt

    thr_key = desired

    # Compact candidates: first all keys > threshold, then keys == threshold.
    tl.store(num_row + tl.arange(0, 2), 0)
    num_ptrs = num_row + tl.zeros([BLOCK_SIZE], tl.int32)

    for t in tl.range(0, n_tiles):
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
        x_key = _convert_to_uint32(x)
        take_gt = in_range & (x_key > thr_key)
        pos = tl.atomic_add(num_ptrs, ones, mask=take_gt)
        tl.store(out_row + pos * stride_outn, offs.to(tl.int32), mask=take_gt & (pos < TOPK))

    for t in tl.range(0, n_tiles):
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
        x_key = _convert_to_uint32(x)
        take_eq = in_range & (x_key == thr_key)
        pos = tl.atomic_add(num_ptrs, ones, mask=take_eq)
        tl.store(out_row + pos * stride_outn, offs.to(tl.int32), mask=take_eq & (pos < TOPK))


# %%
# TLE kernel (shared memory)
# --------------------------


@triton.autotune(
    configs=TLE_TOPK_AUTOTUNE_CONFIGS,
    key=["seq_len"],
)
@triton.jit
def tle_topk_selector_kernel(
    x_ptr,
    out_ptr,
    starts_ptr,
    ends_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    seq_len,
    RADIX: tl.constexpr,
    HIST_SIZE: tl.constexpr,
    ASSUME_ALIGNED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_TILES: tl.constexpr,
    SMEM_INPUT: tl.constexpr,
    NUM_INPUT_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = tl.load(starts_ptr + pid).to(tl.int32)
    row_end = tl.load(ends_ptr + pid).to(tl.int32)

    row_ptr = x_ptr + pid * stride_xm
    out_row = out_ptr + pid * stride_outm

    if ASSUME_ALIGNED:
        tl.assume(row_start == 0)
        tl.assume(row_end == seq_len)
        tl.assume(stride_xn == 1)
        tl.assume(stride_outn == 1)
        seq_len = tl.multiple_of(seq_len, BLOCK_SIZE)

    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)

    s_histogram = tle.gpu.alloc(
        [HIST_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
========
    hist_ptrs = tle.gpu.local_ptr(s_histogram, (bins, ))

    # Ping-pong candidate buffers in shared memory.
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
    s_num_input = tle.gpu.alloc(
        [2],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_input_idx = tle.gpu.alloc(
<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
        [2, SMEM_INPUT],
========
        [2, SMEM_INPUT_SIZE],
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )

<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
    hist_idx = tl.arange(0, RADIX)
    hist_last = tl.full([1], RADIX, tl.int32)

    hist_ptrs = tle.gpu.local_ptr(s_histogram, (hist_idx, ))
    hist_last_ptrs = tle.gpu.local_ptr(s_histogram, (hist_last, ))
    tl.store(hist_ptrs, 0)
    tl.store(hist_last_ptrs, 0)
    tl.store(tle.gpu.local_ptr(s_num_input, (tl.arange(0, 2), )), 0)
    tl.debug_barrier()

    l_new_topk = tl.full((), TOPK, tl.int32)

    # stage 1
    for t in tl.static_range(N_TILES):
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=0.0)
        bin_u16 = _convert_to_uint16(x)
        bin_i32 = bin_u16.to(tl.int32)
        hist_bin_ptrs = tle.gpu.local_ptr(s_histogram, (bin_i32, ))
        tl.atomic_add(hist_bin_ptrs, ones, mask=in_range)

    rev_idx = (RADIX - 1) - hist_idx
    hist_rev = tl.load(tle.gpu.local_ptr(s_histogram, (rev_idx, )))
    hist_cum_rev = tl.cumsum(hist_rev, axis=0)
    tl.store(tle.gpu.local_ptr(s_histogram, (rev_idx, )), hist_cum_rev)
    tl.debug_barrier()

    hist_cum = tl.load(hist_ptrs)
    hist_cum_next = tl.load(tle.gpu.local_ptr(s_histogram, (hist_idx + 1, )), mask=hist_idx + 1 < RADIX, other=0)
    cond = (hist_cum > l_new_topk) & (hist_cum_next <= l_new_topk)
    cand = tl.where(cond, hist_idx.to(tl.int32), -1)
    threshold = tl.max(cand, axis=0)
    hist_next = tl.max(tl.where(hist_idx == threshold + 1, hist_cum, 0), axis=0)
    l_new_topk = tl.maximum(l_new_topk - hist_next, 0)

    num_ptrs = tle.gpu.local_ptr(s_num_input, (tl.zeros([BLOCK_SIZE], tl.int32), ))
    for t in tl.static_range(N_TILES):
========
    # Stage 1: coarse 8-bit prescreen on fp16-mapped keys.
    tl.store(hist_ptrs, tl.zeros([RADIX_SIZE], dtype=tl.int32))
    tl.store(tle.gpu.local_ptr(s_histogram, (RADIX_SIZE, )), 0)
    tl.store(tle.gpu.local_ptr(s_num_input, (0, )), 0)
    tl.store(tle.gpu.local_ptr(s_num_input, (1, )), 0)
    for t in tl.range(0, n_tiles):
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
        digit8 = _convert_to_uint16_hi8(x)
        tl.atomic_add(tle.gpu.local_ptr(s_histogram, (digit8, )), ones, mask=in_range, sem="relaxed", scope="cta")

    counts = tl.load(hist_ptrs)
    cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
    tl.store(hist_ptrs, cumsum_desc)
    tl.store(tle.gpu.local_ptr(s_histogram, (RADIX_SIZE, )), 0)

    # TileLang-style threshold: find bin with cumsum(bin) > k and cumsum(bin+1) <= k.
    new_topk = tl.full((), TOPK, dtype=tl.int32)
    cond = cumsum_desc > new_topk
    threshold_bin = tl.max(tl.where(cond, bins, 0), axis=0).to(tl.int32)
    counts_gt = tl.load(tle.gpu.local_ptr(s_histogram, (threshold_bin + 1, )))
    new_topk = new_topk - counts_gt

    # Stage 2: write coarse winners and cache threshold-bin indices into shared memory.
    num0_ptrs = tle.gpu.local_ptr(s_num_input, (tl.zeros([BLOCK_SIZE], dtype=tl.int32), ))
    for t in tl.range(0, n_tiles):
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=0.0)
        bin_u16 = _convert_to_uint16(x)
        bin_i32 = bin_u16.to(tl.int32)
        gt_thr = bin_i32 > threshold
        eq_thr = bin_i32 == threshold

<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
        pos = tl.atomic_add(tle.gpu.local_ptr(s_histogram, (bin_i32 + 1, )), ones, mask=in_range & gt_thr)
        pos = tl.where(in_range & gt_thr, pos, 0)
        tl.store(out_row + pos * stride_outn, offs.to(tl.int32), mask=in_range & gt_thr & (pos < TOPK))
========
        take_gt = in_range & (digit8 > threshold_bin)
        pos_gt = tl.atomic_add(
            tle.gpu.local_ptr(s_histogram, (digit8 + 1, )),
            ones,
            mask=take_gt,
            sem="relaxed",
            scope="cta",
        )
        tl.store(out_row + pos_gt * stride_outn, offs.to(tl.int32), mask=take_gt & (pos_gt < TOPK))
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py

        pos_eq = tl.atomic_add(num_ptrs, ones, mask=in_range & eq_thr & (l_new_topk > 0))
        pos_eq = tl.where(in_range & eq_thr, pos_eq, 0)
        tl.store(
<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
            tle.gpu.local_ptr(s_input_idx, (tl.zeros([BLOCK_SIZE], tl.int32), pos_eq)),
========
            tle.gpu.local_ptr(s_input_idx, (tl.zeros([BLOCK_SIZE], dtype=tl.int32), pos_eq)),
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
            offs.to(tl.int32),
            mask=in_range & eq_thr & (pos_eq < SMEM_INPUT) & (l_new_topk > 0),
        )

    # stage 2
    for round_id in tl.static_range(4):
        r_idx = round_id & 1
        next_idx = r_idx ^ 1
        start_pos = TOPK - l_new_topk

<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
        tl.store(hist_ptrs, 0)
        tl.store(hist_last_ptrs, 0)
        num_ptrs_next = tle.gpu.local_ptr(s_num_input, (tl.full([BLOCK_SIZE], next_idx, tl.int32), ))
        tl.store(num_ptrs_next, 0, mask=lane == 0)
        tl.debug_barrier()

        num_ptrs_r = tle.gpu.local_ptr(s_num_input, (tl.full([BLOCK_SIZE], r_idx, tl.int32), ))
        l_num_input = tl.max(tl.load(num_ptrs_r), axis=0).to(tl.int32)
        max_input = tl.full((), SMEM_INPUT, tl.int32)
        l_num_input = tl.minimum(l_num_input, max_input)
        active = l_new_topk > 0

        shift = 24 - round_id * 8
        for t in tl.static_range(NUM_INPUT_TILES):
            offs = t * BLOCK_SIZE + lane
            valid = offs < l_num_input
            cand_idx = tl.load(
                tle.gpu.local_ptr(s_input_idx, (tl.full([BLOCK_SIZE], r_idx, tl.int32), offs)),
                mask=valid,
                other=0,
            )
            x = tl.load(row_ptr + cand_idx * stride_xn, mask=valid, other=0.0)
            bin_u32 = _convert_to_uint32(x)
            bin_i32 = ((bin_u32 >> shift) & 0xFF).to(tl.int32)
            tl.atomic_add(tle.gpu.local_ptr(s_histogram, (bin_i32, )), ones, mask=valid & active)

        rev_idx = (RADIX - 1) - hist_idx
        hist_rev = tl.load(tle.gpu.local_ptr(s_histogram, (rev_idx, )))
        hist_cum_rev = tl.cumsum(hist_rev, axis=0)
        tl.store(tle.gpu.local_ptr(s_histogram, (rev_idx, )), hist_cum_rev)
        tl.debug_barrier()

        hist_cum = tl.load(hist_ptrs)
        hist_cum_next = tl.load(tle.gpu.local_ptr(s_histogram, (hist_idx + 1, )), mask=hist_idx + 1 < RADIX, other=0)
        cond = (hist_cum > l_new_topk) & (hist_cum_next <= l_new_topk)
        cand = tl.where(cond, hist_idx.to(tl.int32), -1)
        threshold = tl.max(cand, axis=0)
        hist_next = tl.max(tl.where(hist_idx == threshold + 1, hist_cum, 0), axis=0)
        l_new_topk = tl.maximum(l_new_topk - hist_next, 0)

        for t in tl.static_range(NUM_INPUT_TILES):
            offs = t * BLOCK_SIZE + lane
            valid = offs < l_num_input
            cand_idx = tl.load(
                tle.gpu.local_ptr(s_input_idx, (tl.full([BLOCK_SIZE], r_idx, tl.int32), offs)),
                mask=valid,
                other=0,
            )
            x = tl.load(row_ptr + cand_idx * stride_xn, mask=valid, other=0.0)
            bin_u32 = _convert_to_uint32(x)
            bin_i32 = ((bin_u32 >> shift) & 0xFF).to(tl.int32)

            gt_thr = bin_i32 > threshold
            eq_thr = bin_i32 == threshold
            pos = tl.atomic_add(tle.gpu.local_ptr(s_histogram, (bin_i32 + 1, )), ones, mask=valid & gt_thr & active)
            pos = tl.where(valid & gt_thr & active, pos, 0)
            out_pos = pos + start_pos
            tl.store(
                out_row + out_pos * stride_outn,
                cand_idx,
                mask=valid & gt_thr & active & (out_pos < TOPK),
            )

            if round_id == 3:
                pos_eq = tl.atomic_add(
                    tle.gpu.local_ptr(s_histogram, (bin_i32 + 1, )),
========
            tl.store(hist_ptrs, tl.zeros([RADIX_SIZE], dtype=tl.int32))
            tl.store(tle.gpu.local_ptr(s_histogram, (RADIX_SIZE, )), 0)
            tl.store(tle.gpu.local_ptr(s_num_input, (nxt_buf, )), 0)

            num_in = tl.load(tle.gpu.local_ptr(s_num_input, (cur_buf, )))
            num_in_tiles = tl.cdiv(num_in, BLOCK_SIZE)

            # Histogram over current candidate buffer.
            for t in tl.range(0, num_in_tiles):
                pos = t * BLOCK_SIZE + lane
                valid = pos < num_in
                idx = tl.load(tle.gpu.local_ptr(s_input_idx, (tl.full([BLOCK_SIZE], cur_buf, tl.int32), pos)), mask=valid, other=0)
                x = tl.load(row_ptr + idx * stride_xn, mask=valid, other=float("-inf"))
                key = _convert_to_uint32(x)
                digit = ((key >> shift) & RADIX_MASK).to(tl.int32)
                tl.atomic_add(
                    tle.gpu.local_ptr(s_histogram, (digit, )),
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
                    ones,
                    mask=valid & eq_thr & active & (l_new_topk > 0),
                )
                pos_eq = tl.where(valid & eq_thr & active, pos_eq, 0)
                out_pos = pos_eq + start_pos
                tl.store(
                    out_row + out_pos * stride_outn,
                    cand_idx,
                    mask=valid & eq_thr & active & (out_pos < TOPK) & (l_new_topk > 0),
                )
            else:
                num_ptrs = tle.gpu.local_ptr(s_num_input, (tl.full([BLOCK_SIZE], next_idx, tl.int32), ))
                pos_eq = tl.atomic_add(num_ptrs, ones, mask=valid & eq_thr & active & (l_new_topk > 0))
                pos_eq = tl.where(valid & eq_thr & active, pos_eq, 0)
                tl.store(
                    tle.gpu.local_ptr(s_input_idx, (tl.full([BLOCK_SIZE], next_idx, tl.int32), pos_eq)),
                    cand_idx,
                    mask=valid & eq_thr & active & (pos_eq < SMEM_INPUT) & (l_new_topk > 0),
                )

<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
========
            counts = tl.load(hist_ptrs)
            cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
            tl.store(hist_ptrs, cumsum_desc)
            tl.store(tle.gpu.local_ptr(s_histogram, (RADIX_SIZE, )), 0)

            cond = cumsum_desc > new_topk
            threshold_bin = tl.max(tl.where(cond, bins, 0), axis=0).to(tl.int32)
            counts_gt = tl.load(tle.gpu.local_ptr(s_histogram, (threshold_bin + 1, )))
            new_topk = new_topk - counts_gt

            # Partition candidates: winners to output, threshold equals to next buffer.
            nxt_ptrs = tle.gpu.local_ptr(s_num_input, (tl.full([BLOCK_SIZE], nxt_buf, tl.int32), ))
            for t in tl.range(0, num_in_tiles):
                pos = t * BLOCK_SIZE + lane
                valid = pos < num_in
                idx = tl.load(tle.gpu.local_ptr(s_input_idx, (tl.full([BLOCK_SIZE], cur_buf, tl.int32), pos)), mask=valid, other=0)
                x = tl.load(row_ptr + idx * stride_xn, mask=valid, other=float("-inf"))
                key = _convert_to_uint32(x)
                digit = ((key >> shift) & RADIX_MASK).to(tl.int32)

                take_gt = valid & (digit > threshold_bin)
                pos_gt = tl.atomic_add(
                    tle.gpu.local_ptr(s_histogram, (digit + 1, )),
                    ones,
                    mask=take_gt,
                    sem="relaxed",
                    scope="cta",
                )
                out_pos_gt = pos_gt + start_pos
                tl.store(out_row + out_pos_gt * stride_outn, idx, mask=take_gt & (out_pos_gt < TOPK))

                take_eq = valid & (digit == threshold_bin) & (new_topk > 0)
                if round_idx == 3:
                    pos_eq = tl.atomic_add(
                        tle.gpu.local_ptr(s_histogram, (digit + 1, )),
                        ones,
                        mask=take_eq,
                        sem="relaxed",
                        scope="cta",
                    )
                    out_pos_eq = pos_eq + start_pos
                    tl.store(out_row + out_pos_eq * stride_outn, idx, mask=take_eq & (out_pos_eq < TOPK))
                else:
                    nxt_pos = tl.atomic_add(nxt_ptrs, ones, mask=take_eq, sem="relaxed", scope="cta")
                    tl.store(
                        tle.gpu.local_ptr(s_input_idx, (tl.full([BLOCK_SIZE], nxt_buf, tl.int32), nxt_pos)),
                        idx,
                        mask=take_eq & (nxt_pos < SMEM_INPUT_SIZE),
                    )


@triton.jit
def triton_topk_selector_kernel(
    x_ptr,
    out_ptr,
    cand0_ptr,
    cand1_ptr,
    starts_ptr,
    ends_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    stride_c0m,
    stride_c0n,
    stride_c1m,
    stride_c1n,
    seq_len,
    ASSUME_ALIGNED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RADIX_BITS: tl.constexpr,
):
    tl.static_assert(RADIX_BITS == 8, "triton_topk_selector_kernel currently expects 8-bit radix")
    pid = tl.program_id(0)
    row_start = tl.load(starts_ptr + pid).to(tl.int32)
    row_end = tl.load(ends_ptr + pid).to(tl.int32)
    row_ptr = x_ptr + pid * stride_xm
    out_row = out_ptr + pid * stride_outm
    cand0_row = cand0_ptr + pid * stride_c0m
    cand1_row = cand1_ptr + pid * stride_c1m

    if ASSUME_ALIGNED:
        tl.assume(row_start == 0)
        tl.assume(row_end == seq_len)
        tl.assume(stride_xn == 1)
        tl.assume(stride_outn == 1)
        seq_len = tl.multiple_of(seq_len, BLOCK_SIZE)

    lane = tl.arange(0, BLOCK_SIZE)
    n_tiles = tl.cdiv(seq_len, BLOCK_SIZE)
    RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
    RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
    bins = tl.arange(0, RADIX_SIZE)

    # Stage 1: 8-bit coarse prescreen on fp16-mapped keys.
    coarse_counts = tl.zeros([RADIX_SIZE], dtype=tl.int32)
    for t in tl.range(0, n_tiles):
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
        digit8 = _convert_to_uint16_hi8(x)
        coarse_counts = coarse_counts + tl.histogram(digit8, RADIX_SIZE, mask=in_range)

    coarse_cumsum_desc = tl.cumsum(coarse_counts, axis=0, reverse=True)
    topk_target = tl.full((), TOPK, tl.int32)
    coarse_cond = coarse_cumsum_desc > topk_target
    coarse_threshold_bin = tl.max(tl.where(coarse_cond, bins, 0), axis=0).to(tl.int32)
    coarse_counts_gt = tl.max(tl.where(bins == (coarse_threshold_bin + 1), coarse_cumsum_desc, 0), axis=0)
    new_topk = topk_target - coarse_counts_gt
    write_count = tl.full((), 0, tl.int32)
    cand_count0 = tl.full((), 0, tl.int32)

    # Stage 2: write coarse winners and compact coarse-threshold candidates into cand0.
    for t in tl.range(0, n_tiles):
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
        digit8 = _convert_to_uint16_hi8(x)
        take = in_range & (digit8 > coarse_threshold_bin)
        take_i32 = take.to(tl.int32)
        pos = write_count + tl.cumsum(take_i32, axis=0) - 1
        mask = take & (pos < TOPK)
        tl.store(out_row + pos * stride_outn, offs.to(tl.int32), mask=mask)
        write_count = write_count + tl.sum(take_i32, axis=0)

        take_eq = in_range & (digit8 == coarse_threshold_bin)
        take_eq_i32 = take_eq.to(tl.int32)
        pos_eq = cand_count0 + tl.cumsum(take_eq_i32, axis=0) - 1
        tl.store(cand0_row + pos_eq * stride_c0n, offs.to(tl.int32), mask=take_eq)
        cand_count0 = cand_count0 + tl.sum(take_eq_i32, axis=0)

    # Stage 3: four 8-bit refinements over compact candidate lists.
    num_in = cand_count0
    for round_idx in tl.static_range(4):
        if (new_topk > 0) & (num_in > 0):
            shift: tl.constexpr = 24 - round_idx * 8
            desired = tl.full((), 0, tl.uint32)
            desired_mask = tl.full((), 0, tl.uint32)
            k_to_find = new_topk
            num_in_tiles = tl.cdiv(num_in, BLOCK_SIZE)
            counts = tl.zeros([RADIX_SIZE], dtype=tl.int32)

            # Histogram on current candidate table.
            for t in tl.range(0, num_in_tiles):
                pos = t * BLOCK_SIZE + lane
                valid = pos < num_in
                if round_idx & 1:
                    idx = tl.load(cand1_row + pos * stride_c1n, mask=valid, other=0)
                else:
                    idx = tl.load(cand0_row + pos * stride_c0n, mask=valid, other=0)
                x = tl.load(row_ptr + idx * stride_xn, mask=valid, other=float("-inf"))
                x_key = _convert_to_uint32(x)
                matches = (x_key & desired_mask) == desired
                take = valid & matches
                digit = ((x_key >> shift) & RADIX_MASK).to(tl.int32)
                counts = counts + tl.histogram(digit, RADIX_SIZE, mask=take)

            cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
            cond = cumsum_desc > k_to_find
            threshold_bin = tl.max(tl.where(cond, bins, 0), axis=0).to(tl.int32)
            counts_gt = tl.max(tl.where(bins == (threshold_bin + 1), cumsum_desc, 0), axis=0)
            desired = desired | (threshold_bin.to(tl.uint32) << shift)
            desired_mask = desired_mask | (tl.full((), RADIX_MASK, tl.uint32) << shift)
            new_topk = k_to_find - counts_gt

            out_count = write_count
            next_count = tl.full((), 0, tl.int32)
            for t in tl.range(0, num_in_tiles):
                pos = t * BLOCK_SIZE + lane
                valid = pos < num_in
                if round_idx & 1:
                    idx = tl.load(cand1_row + pos * stride_c1n, mask=valid, other=0)
                else:
                    idx = tl.load(cand0_row + pos * stride_c0n, mask=valid, other=0)
                x = tl.load(row_ptr + idx * stride_xn, mask=valid, other=float("-inf"))
                x_key = _convert_to_uint32(x)
                digit = ((x_key >> shift) & RADIX_MASK).to(tl.int32)

                take_gt = valid & (digit > threshold_bin)
                take_gt_i32 = take_gt.to(tl.int32)
                out_pos_gt = out_count + tl.cumsum(take_gt_i32, axis=0) - 1
                out_mask_gt = take_gt & (out_pos_gt < TOPK)
                tl.store(out_row + out_pos_gt * stride_outn, idx, mask=out_mask_gt)
                out_count = out_count + tl.sum(take_gt_i32, axis=0)

                take_eq = valid & (digit == threshold_bin)
                take_eq_i32 = take_eq.to(tl.int32)
                if round_idx == 3:
                    out_pos_eq = out_count + tl.cumsum(take_eq_i32, axis=0) - 1
                    out_mask_eq = take_eq & (out_pos_eq < TOPK)
                    tl.store(out_row + out_pos_eq * stride_outn, idx, mask=out_mask_eq)
                    out_count = out_count + tl.sum(take_eq_i32, axis=0)
                else:
                    nxt_pos = next_count + tl.cumsum(take_eq_i32, axis=0) - 1
                    if round_idx & 1:
                        tl.store(cand0_row + nxt_pos * stride_c0n, idx, mask=take_eq)
                    else:
                        tl.store(cand1_row + nxt_pos * stride_c1n, idx, mask=take_eq)
                    next_count = next_count + tl.sum(take_eq_i32, axis=0)

            write_count = out_count
            num_in = next_count
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py

# %%
# TileLang reference (optional)
# -----------------------------

if _HAVE_TILELANG:
    _TL_PASS_CONFIGS = {
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
    _TL_KERNEL_CACHE = {}

    def convert_to_uint16(x):
        hval = T.Cast(T.float16, x)
        bits_uint = T.reinterpret(T.uint16, hval)
        bits_uint = T.if_then_else(x < 0, ~bits_uint & 0xFFFF, bits_uint | 0x8000)
        return bits_uint >> 8

    def convert_to_uint32(x):
        bits_uint = T.reinterpret(T.uint32, x)
        bits_uint = T.if_then_else(
            x < 0,
            ~bits_uint & T.Cast(T.uint32, 0xFFFFFFFF),
            bits_uint | T.Cast(T.uint32, 0x80000000),
        )
        return bits_uint

    @tilelang.jit(pass_configs=_TL_PASS_CONFIGS)
    def _tilelang_topk_impl(topk, in_dtype=T.float32, out_dtype=T.int32):
        batch = T.dynamic("batch")
        seq_len = T.dynamic("seq_len")
        RADIX_LOCAL = 1 << 8
        BLOCK_SIZE = 1024
        SMEM_INPUT_SIZE = 4096

        @T.prim_func
        def tl_topk_kernel(
            input: T.Tensor[(batch, seq_len), in_dtype],
            index: T.Tensor[(batch, topk), out_dtype],
            starts: T.Tensor[(batch), out_dtype],
            ends: T.Tensor[(batch), out_dtype],
        ):
            with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
                tx = T.get_thread_binding()

                s_threshold_bin_id = T.alloc_shared([1], T.int32)
                s_histogram = T.alloc_shared([RADIX_LOCAL + 1], T.int32)
                s_num_input = T.alloc_shared([2], T.int32)
                s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

                l_threshold_bin_id = T.alloc_var(T.int32)
                l_new_topk = T.alloc_var(T.int32)
                l_num_input = T.alloc_var(T.int32)
                l_bin_id32 = T.alloc_var(T.int32)
                l_val = T.alloc_var(T.int32)
                l_start_pos = T.alloc_var(T.int32)
                l_start_idx = T.alloc_var(T.int32)
                l_end_idx = T.alloc_var(T.int32)
                l_out_pos = T.alloc_var(T.int32)

                l_new_topk = topk
                l_start_idx = starts[bx]
                l_end_idx = ends[bx]

                T.fill(s_histogram, 0)
                T.fill(s_num_input[0], 0)
                T.sync_threads()
                for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                    input_idx = s * BLOCK_SIZE + tx
                    if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                        inval_int16 = convert_to_uint16(input[bx, input_idx])
                        T.atomic_add(s_histogram[inval_int16], 1)
                T.sync_threads()

                if tx < RADIX_LOCAL:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX_LOCAL)
                        if tx < RADIX_LOCAL - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX_LOCAL)
                        if tx < RADIX_LOCAL - offset:
                            s_histogram[tx] = l_val

                    T.sync_threads(3, RADIX_LOCAL)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()
                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                    T.sync_threads()
                    input_idx = s * BLOCK_SIZE + tx
                    if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                        bin_id = convert_to_uint16(input[bx, input_idx])
                        l_bin_id32 = T.Cast(T.int32, bin_id)
                        if l_bin_id32 > l_threshold_bin_id:
                            pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                            index[bx, pos] = input_idx
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                            s_input_idx[0, pos] = input_idx

                for round in T.serial(4):
                    if l_new_topk <= 0:
                        T.loop_break()

                    r_idx = round % 2
                    l_start_pos = topk - l_new_topk

                    T.sync_threads()
                    T.fill(s_histogram, 0)
                    if tx == 0:
                        s_num_input[r_idx ^ 1] = 0
                    T.sync_threads()

                    l_num_input = s_num_input[r_idx]
                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        if s * BLOCK_SIZE + tx < l_num_input:
                            l_bin_id32 = T.Cast(
                                T.int32,
                                ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                                  (24 - round * 8)) & 0xFF),
                            )
                            T.atomic_add(s_histogram[l_bin_id32], 1)
                    T.sync_threads()

                    if tx < RADIX_LOCAL:
                        for i in T.serial(8):
                            offset = 1 << i
                            T.sync_threads(3, RADIX_LOCAL)
                            if tx < RADIX_LOCAL - offset:
                                l_val = s_histogram[tx] + s_histogram[tx + offset]
                            T.sync_threads(3, RADIX_LOCAL)
                            if tx < RADIX_LOCAL - offset:
                                s_histogram[tx] = l_val

                        T.sync_threads(3, RADIX_LOCAL)
                        if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                            s_threshold_bin_id[0] = tx
                    T.sync_threads()

                    l_threshold_bin_id = s_threshold_bin_id[0]
                    l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                    T.sync_threads()

                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        T.sync_threads()
                        if s * BLOCK_SIZE + tx < l_num_input:
                            l_bin_id32 = T.Cast(
                                T.int32,
                                ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                                  (24 - round * 8)) & 0xFF),
                            )
                            if l_bin_id32 > l_threshold_bin_id:
                                pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                index[bx, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                if round == 3:
                                    l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1,
                                                             return_prev=True) + l_start_pos
                                    if l_out_pos < topk:
                                        index[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                else:
                                    pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                    s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

        return tl_topk_kernel

    def tilelang_topk_selector(input, starts, ends, topk, out: Optional[torch.Tensor] = None):
        batch, _ = input.shape
        if out is None:
            out = torch.zeros((batch, topk), dtype=torch.int32, device=input.device)
        kernel = _TL_KERNEL_CACHE.get(topk)
        if kernel is None:
            kernel = _tilelang_topk_impl(topk)
            _TL_KERNEL_CACHE[topk] = kernel
        kernel(input, out, starts, ends)
        return out


# %%
# Python wrappers
# ---------------


def _allocate_triton_scratch(batch, smem_input, device):
    hist = torch.empty((batch, RADIX + 1), dtype=torch.int32, device=device)
    num = torch.empty((batch, 2), dtype=torch.int32, device=device)
    return hist, num


def triton_topk_selector(
    x,
    starts,
    ends,
    topk,
    block_size=1024,
    num_warps=32,
    smem_input=4096,
    out: Optional[torch.Tensor] = None,
    scratch=None,
    assume_aligned: Optional[bool] = None,
):
    if x.dtype != torch.float32:
        x = x.float()
    batch, seq_len = x.shape
    if out is None:
        out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
    if scratch is None:
        scratch = _allocate_triton_scratch(batch, smem_input, x.device)
    if len(scratch) == 3:
        hist, num, _ = scratch
    else:
        hist, num = scratch

    if assume_aligned is None:
        assume_aligned = (x.is_contiguous() and out.is_contiguous() and (seq_len % block_size == 0)
                          and torch.all(starts == 0).item() and torch.all(ends == seq_len).item())

    grid = (batch, )
    triton_topk_selector_kernel[grid](
        x,
        out,
        starts,
        ends,
        hist,
        num,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        hist.stride(0),
        num.stride(0),
        seq_len,
        RADIX_BITS=8,
        ASSUME_ALIGNED=assume_aligned,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out


def tle_topk_selector(
    x,
    starts,
    ends,
    topk,
    block_size=1024,
<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
    num_warps=32,
    smem_input=4096,
========
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
    out: Optional[torch.Tensor] = None,
    assume_aligned: Optional[bool] = None,
):
    if x.dtype != torch.float32:
        x = x.float()
    batch, seq_len = x.shape
    if out is None:
        out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)

    n_tiles = triton.cdiv(seq_len, block_size)
    num_input_tiles = triton.cdiv(smem_input, block_size)
    hist_size = RADIX * 2
    if assume_aligned is None:
        assume_aligned = (x.is_contiguous() and out.is_contiguous() and (seq_len % block_size == 0)
                          and torch.all(starts == 0).item() and torch.all(ends == seq_len).item())

    batch, seq_len = x.shape
    grid = (batch, )
    tle_topk_selector_kernel[grid](
        x,
        out,
        starts,
        ends,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        seq_len,
        RADIX=RADIX,
        HIST_SIZE=hist_size,
        ASSUME_ALIGNED=assume_aligned,
        TOPK=topk,
        BLOCK_SIZE=block_size,
<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
        N_TILES=n_tiles,
        SMEM_INPUT=smem_input,
        NUM_INPUT_TILES=num_input_tiles,
        num_warps=num_warps,
========
        SMEM_INPUT_SIZE=4096,
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
    )
    return out


<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
========
def triton_topk_selector(
    x,
    starts,
    ends,
    topk,
    block_size=1024,
    out: Optional[torch.Tensor] = None,
    cand0: Optional[torch.Tensor] = None,
    cand1: Optional[torch.Tensor] = None,
    assume_aligned: Optional[bool] = None,
):
    if x.dtype != torch.float32:
        x = x.float()
    batch, seq_len = x.shape
    if out is None:
        out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
    if cand0 is None:
        cand0 = torch.empty((batch, seq_len), dtype=torch.int32, device=x.device)
    if cand1 is None:
        cand1 = torch.empty((batch, seq_len), dtype=torch.int32, device=x.device)

    if assume_aligned is None:
        assume_aligned = (
            x.is_contiguous()
            and out.is_contiguous()
            and (seq_len % block_size == 0)
            and torch.all(starts == 0).item()
            and torch.all(ends == seq_len).item()
        )

    assert cand0.shape == (batch, seq_len) and cand0.dtype == torch.int32 and cand0.is_cuda
    assert cand1.shape == (batch, seq_len) and cand1.dtype == torch.int32 and cand1.is_cuda

    # Triton kernel uses kernel-specific tuning to avoid slow/unstable configs.
    kernel_num_warps = 4 if block_size >= 1024 else 8

    grid = (batch, )
    triton_topk_selector_kernel[grid](
        x,
        out,
        cand0,
        cand1,
        starts,
        ends,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        cand0.stride(0),
        cand0.stride(1),
        cand1.stride(0),
        cand1.stride(1),
        seq_len,
        ASSUME_ALIGNED=assume_aligned,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        RADIX_BITS=8,
        num_warps=kernel_num_warps,
        num_stages=1,
    )
    return out


>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
# %%
# Correctness & benchmarking
# --------------------------


def _torch_topk_indices(x, starts, ends, topk):
    batch, _ = x.shape
    out = torch.empty((batch, topk), dtype=torch.int32, device=x.device)
    for i in range(batch):
        start = int(starts[i].item())
        end = int(ends[i].item())
        vals, idx = torch.topk(x[i, start:end], topk, dim=0)
        out[i] = idx.to(torch.int32) + start
    return out


def _recall(pred, ref):
    batch = pred.shape[0]
    k = ref.shape[1]
    hits = 0
    for i in range(batch):
        pred_set = set(pred[i].tolist())
        ref_set = set(ref[i].tolist())
        hits += len(pred_set & ref_set)
    return hits / (batch * k)


_BENCH_PROVIDERS = ["triton", "tle", "torch"] + (["tilelang"] if _HAVE_TILELANG else [])
_BENCH_NAMES = ["Triton-Radix", "TLE-DeepSeek", "Torch-TopK"] + (["TileLang"] if _HAVE_TILELANG else [])
_BENCH_STYLES = [("red", "-"), ("orange", "-"), ("green", "-")] + ([("blue", "-")] if _HAVE_TILELANG else [])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch", "seq_len", "topk"],
        x_vals=[
            (64, 4096, 128),
            (64, 8192, 256),
            (64, 32768, 1024),
            (64, 32768, 2048),
        ],
        x_log=True,
        line_arg="provider",
        line_vals=_BENCH_PROVIDERS,
        line_names=_BENCH_NAMES,
        styles=_BENCH_STYLES,
        ylabel="ms",
        plot_name="tle-deepseek-v32-topk-selector",
        args={},
<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
    ))
def benchmark(batch, seq_len, topk, provider, block_size, smem_input, num_warps, warmup, rep):
    if topk > smem_input:
        return float("nan"), float("nan"), float("nan")

========
    )
)
def benchmark(batch, seq_len, topk, provider, block_size, warmup, rep):
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
    torch.manual_seed(1)
    x = torch.randn(batch, seq_len, device=DEVICE, dtype=torch.float32)
    starts = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    ends = torch.full((batch, ), seq_len, dtype=torch.int32, device=DEVICE)
    assume_aligned = (seq_len % block_size == 0)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        triton_scratch = _allocate_triton_scratch(batch, smem_input, x.device)
        triton_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)

        def run():
            triton_topk_selector(
                x,
                starts,
                ends,
                topk,
                block_size=block_size,
                num_warps=num_warps,
                smem_input=smem_input,
                out=triton_out,
                scratch=triton_scratch,
                assume_aligned=assume_aligned,
            )

    elif provider == "tle":
        tle_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)

        def run():
            tle_topk_selector(
                x,
                starts,
                ends,
                topk,
                block_size=block_size,
<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
                num_warps=num_warps,
                smem_input=smem_input,
========
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
                out=tle_out,
                assume_aligned=assume_aligned,
            )

<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
========
    elif provider == "triton":
        triton_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
        triton_cand0 = torch.empty((batch, seq_len), dtype=torch.int32, device=x.device)
        triton_cand1 = torch.empty((batch, seq_len), dtype=torch.int32, device=x.device)

        def run():
            triton_topk_selector(
                x,
                starts,
                ends,
                topk,
                block_size=block_size,
                out=triton_out,
                cand0=triton_cand0,
                cand1=triton_cand1,
                assume_aligned=assume_aligned,
            )

>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
    elif provider == "torch":

        def run():
            torch.topk(x, topk, dim=-1)[1]

    else:
        if not _HAVE_TILELANG:
            return float("nan"), float("nan"), float("nan")
        tilelang_out = torch.zeros((batch, topk), dtype=torch.int32, device=x.device)

        def run():
            tilelang_topk_selector(x, starts, ends, topk, out=tilelang_out)

    ms, min_ms, max_ms = triton.testing.do_bench(
        run,
        quantiles=quantiles,
        warmup=warmup,
        rep=rep,
    )
    return ms, max_ms, min_ms


<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
def run_correctness(batch, seq_len, topk, block_size, smem_input, num_warps):
========
def run_correctness(batch, seq_len, topk, block_size):
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
    torch.manual_seed(1)
    x = torch.randn(batch, seq_len, device=DEVICE, dtype=torch.float32)
    starts = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    ends = torch.full((batch, ), seq_len, dtype=torch.int32, device=DEVICE)
    assume_aligned = (seq_len % block_size == 0)

    ref = _torch_topk_indices(x, starts, ends, topk)

<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
========
    tle_out = tle_topk_selector(
        x,
        starts,
        ends,
        topk,
        block_size=block_size,
        assume_aligned=assume_aligned,
    )

    print(f"TLE recall vs torch.topk: {_recall(tle_out, ref):.4f}")
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
    triton_out = triton_topk_selector(
        x,
        starts,
        ends,
        topk,
        block_size=block_size,
<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
        num_warps=num_warps,
        smem_input=smem_input,
========
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
        assume_aligned=assume_aligned,
    )
    tle_out = tle_topk_selector(
        x,
        starts,
        ends,
        topk,
        block_size=block_size,
        num_warps=num_warps,
        smem_input=smem_input,
        assume_aligned=assume_aligned,
    )

    print(f"Triton recall vs torch.topk: {_recall(triton_out, ref):.4f}")
    print(f"TLE recall vs torch.topk: {_recall(tle_out, ref):.4f}")

    if _HAVE_TILELANG:
        tilelang_out = tilelang_topk_selector(x, starts, ends, topk)
        print(f"TileLang recall vs torch.topk: {_recall(tilelang_out, ref):.4f}")
        print(f"Triton recall vs TileLang: {_recall(triton_out, tilelang_out):.4f}")
        print(f"TLE recall vs TileLang: {_recall(tle_out, tilelang_out):.4f}")
    else:
        print("TileLang not available; skipping TileLang correctness.")


<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
def run_bench(block_size, smem_input, num_warps, warmup, rep, show_plots):
========
def run_bench(block_size, warmup, rep, show_plots):
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
    benchmark.run(
        print_data=True,
        show_plots=show_plots,
        block_size=block_size,
<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
        smem_input=smem_input,
        num_warps=num_warps,
========
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
        warmup=warmup,
        rep=rep,
    )


# %%
# Main
# ----


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    parser.add_argument("--seq_len", type=int, default=4096, help="sequence length")
    parser.add_argument("--topk", type=int, default=128, help="top-k")
    parser.add_argument("--block_size", type=int, default=1024, help="block size (threads)")
<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
    parser.add_argument("--smem_input", type=int, default=4096, help="candidate buffer size")
    parser.add_argument("--num_warps", type=int, default=32, help="num warps")
========
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
    parser.add_argument("--warmup", type=int, default=5, help="warmup iters")
    parser.add_argument("--rep", type=int, default=20, help="benchmark iters")
    parser.add_argument("--show_plots", action="store_true", help="show plots in benchmark")
    parser.add_argument("--skip_correctness", action="store_true", help="skip correctness check")
    parser.add_argument("--skip_bench", action="store_true", help="skip benchmark")
    args = parser.parse_args(argv)

    if args.topk > args.smem_input:
        raise ValueError("topk must be <= smem_input to avoid truncation")

    if not args.skip_correctness:
        run_correctness(
            batch=args.batch,
            seq_len=args.seq_len,
            topk=args.topk,
            block_size=args.block_size,
<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
            smem_input=args.smem_input,
            num_warps=args.num_warps,
========
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
        )

    if not args.skip_bench:
        run_bench(
            block_size=args.block_size,
<<<<<<<< HEAD:python/tutorials/tle/05-deepseek_v32_topk_selector.py
            smem_input=args.smem_input,
            num_warps=args.num_warps,
========
>>>>>>>> e74208c96 (WIP: sync local TLE changes):python/tutorials/tle/deepseek_v32/01-topk_selector.py
            warmup=args.warmup,
            rep=args.rep,
            show_plots=args.show_plots,
        )


if __name__ == "__main__":
    main()
