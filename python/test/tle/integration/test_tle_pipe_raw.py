# flagtree tle
"""Known-failure repros for combining ``tle.pipe`` with CUDA ``tle_raw.call``.

The kernels intentionally keep all raw operations synchronous: no TMA,
cp.async, WGMMA, or CTA-wide barrier is used. This isolates problems caused by
the raw region being opaque to pipe lowering, CUDA thread ids retaining CTA
semantics inside worker partitions, and raw code interpreting encoded shared
memory as an ordinary strided memref.

Do not add ``__syncthreads()`` or an intentionally wrong mbarrier protocol to
these default tests. Such a repro can hang the pytest process and GPU context.
"""

from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect


PIPE_CAPACITY = 2
ITERATIONS = 6
ROWS = 8
COLS = 32
FIELDS = 2
SENTINEL = -777.0
RAW_CUDA_FILE = Path(__file__).with_name("tle_pipe_raw_repro.cu")


@dialect(
    name="cuda",
    file=RAW_CUDA_FILE,
    extern_func_name="raw_pipe_producer",
    deferred=True,
)
def _raw_pipe_producer(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    file=RAW_CUDA_FILE,
    extern_func_name="raw_pipe_consumer",
    deferred=True,
)
def _raw_pipe_consumer(*args, **kwargs):
    ...


def _has_hopper_raw_cuda() -> bool:
    try:
        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "cuda" or not torch.cuda.is_available():
            return False
        if torch.cuda.get_device_capability()[0] < 9:
            return False
        # Deferred raw compilation still needs clang when make_llir runs.
        from triton.experimental.tle.raw.cuda.runtime import _resolve_clang
        _resolve_clang()
        return True
    except (ImportError, OSError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _has_hopper_raw_cuda(),
    reason="tle.pipe + CUDA tle_raw repros require NVIDIA Hopper and raw clang",
)


@triton.jit
def _expected_value(iteration, field, row, col):
    return iteration * 100000 + field * 10000 + row * 100 + col


@triton.jit
def _tile_indices(ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.broadcast_to(tl.arange(0, ROWS)[:, None], (ROWS, COLS))
    cols = tl.broadcast_to(tl.arange(0, COLS)[None, :], (ROWS, COLS))
    return rows, cols


@triton.jit
def _store_pipe_slot_to_output(slot, output, iteration, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows, cols = _tile_indices(ROWS, COLS)
    linear = rows * COLS + cols
    iteration_base = iteration * FIELDS * ROWS * COLS
    plain = tl.load(tle.gpu.local_ptr(slot.plain, (rows, cols)))
    swizzled = tl.load(tle.gpu.local_ptr(slot.swizzled, (rows, cols)))
    tl.store(output + iteration_base + linear, plain)
    tl.store(output + iteration_base + ROWS * COLS + linear, swizzled)


@triton.jit
def _case1_consumer(reader, output, ITERATIONS: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    for iteration in tl.range(0, ITERATIONS):
        result = reader.wait(iteration)
        _store_pipe_slot_to_output(result.slot, output, iteration, ROWS, COLS)
        reader.release(iteration)


@triton.jit
def _case1_raw_producer(writer, ITERATIONS: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows, cols = _tile_indices(ROWS, COLS)
    sentinel = tl.full((ROWS, COLS), SENTINEL, tl.float32)
    for iteration in tl.range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        # Make partial raw coverage deterministic instead of reading undefined
        # shared memory from worker lanes that never touch the tile.
        tl.store(tle.gpu.local_ptr(slot.plain, (rows, cols)), sentinel)
        tl.store(tle.gpu.local_ptr(slot.swizzled, (rows, cols)), sentinel)
        tle_raw.call(
            _raw_pipe_producer,
            [slot.plain, slot.swizzled, iteration],
            output_indices=(),
        )
        writer.commit(iteration)


@triton.jit
def _case2_producer(writer, ITERATIONS: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows, cols = _tile_indices(ROWS, COLS)
    for iteration in tl.range(0, ITERATIONS):
        slot = writer.acquire(iteration)
        plain = _expected_value(iteration, 0, rows, cols).to(tl.float32)
        swizzled = _expected_value(iteration, 1, rows, cols).to(tl.float32)
        tl.store(tle.gpu.local_ptr(slot.plain, (rows, cols)), plain)
        tl.store(tle.gpu.local_ptr(slot.swizzled, (rows, cols)), swizzled)
        writer.commit(iteration)


@triton.jit
def _case2_raw_consumer(reader, output, ITERATIONS: tl.constexpr):
    for iteration in tl.range(0, ITERATIONS):
        result = reader.wait(iteration)
        tle_raw.call(
            _raw_pipe_consumer,
            [result.slot.plain, result.slot.swizzled, output, iteration],
            output_indices=(),
        )
        reader.release(iteration)


@triton.jit
def _case1_producer_raw_kernel(
    output,
    ITERATIONS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    PIPE_CAPACITY: tl.constexpr,
):
    plain = tle.gpu.alloc(
        [PIPE_CAPACITY, ROWS, COLS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    swizzled = tle.gpu.alloc(
        [PIPE_CAPACITY, ROWS, COLS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
    )
    pipe = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="producer_raw", plain=plain, swizzled=swizzled)

    # Consumer is the default partition. Producer is deliberately a worker so
    # CUDA CTA-global thread indices differ from partition-local expectations.
    tle.gpu.warp_specialize(
        [
            (_case1_consumer, (pipe.reader(), output, ITERATIONS, ROWS, COLS)),
            (_case1_raw_producer, (pipe.writer(), ITERATIONS, ROWS, COLS)),
        ],
        [4],
        [64],
    )


@triton.jit
def _case2_consumer_raw_kernel(
    output,
    ITERATIONS: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    PIPE_CAPACITY: tl.constexpr,
):
    plain = tle.gpu.alloc(
        [PIPE_CAPACITY, ROWS, COLS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    swizzled = tle.gpu.alloc(
        [PIPE_CAPACITY, ROWS, COLS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
    )
    pipe = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="consumer_raw", plain=plain, swizzled=swizzled)

    # Producer is the default partition; raw consumer is the worker partition.
    tle.gpu.warp_specialize(
        [
            (_case2_producer, (pipe.writer(), ITERATIONS, ROWS, COLS)),
            (_case2_raw_consumer, (pipe.reader(), output, ITERATIONS)),
        ],
        [4],
        [64],
    )


def _reference() -> torch.Tensor:
    iteration = torch.arange(ITERATIONS, dtype=torch.float32)[:, None, None, None]
    field = torch.arange(FIELDS, dtype=torch.float32)[None, :, None, None]
    row = torch.arange(ROWS, dtype=torch.float32)[None, None, :, None]
    col = torch.arange(COLS, dtype=torch.float32)[None, None, None, :]
    return iteration * 100000 + field * 10000 + row * 100 + col


def _launch(kernel):
    assert ITERATIONS > PIPE_CAPACITY
    # Each slot is used three times, producing two phase flips per slot.
    assert ITERATIONS >= PIPE_CAPACITY * 3
    output = torch.full(
        (ITERATIONS, FIELDS, ROWS, COLS),
        SENTINEL,
        device="cuda",
        dtype=torch.float32,
    )
    compiled = kernel[(1, )](
        output,
        ITERATIONS=ITERATIONS,
        ROWS=ROWS,
        COLS=COLS,
        PIPE_CAPACITY=PIPE_CAPACITY,
        num_warps=8,
    )
    torch.cuda.synchronize()
    return output.cpu(), compiled


def _assert_lowering_boundary(compiled):
    ttgir = compiled.asm["ttgir"]
    assert "tle.pipe" not in ttgir
    assert "tle.dsl_region" in ttgir
    assert any(op in ttgir for op in ("nvws.", "ttng.wait_barrier", "ttng.arrive_barrier"))


def _mismatch_diagnostic(actual: torch.Tensor, expected: torch.Tensor) -> str:
    mismatch = actual != expected
    sentinel_count = int((actual == SENTINEL).sum().item())
    plain_errors = int(mismatch[:, 0].sum().item())
    swizzled_errors = int(mismatch[:, 1].sum().item())
    failing = mismatch.nonzero()
    if failing.numel():
        iteration, field, row, col = (int(value) for value in failing[0].tolist())
        first = (
            f"iteration={iteration}, stage={iteration % PIPE_CAPACITY}, "
            f"field={field}, row={row}, col={col}"
        )
    else:
        first = None
    return (
        f"sentinel_count={sentinel_count}, plain_errors={plain_errors}, "
        f"swizzled_errors={swizzled_errors}, first_error={first}"
    )


@pytest.mark.parametrize("kernel", [_case1_producer_raw_kernel, _case2_consumer_raw_kernel])
def test_pipe_raw_lowering_boundary(kernel):
    _, compiled = _launch(kernel)
    _assert_lowering_boundary(compiled)


@pytest.mark.xfail(
    strict=True,
    reason="raw producer uses CTA-global thread ids and ordinary strides in a worker partition",
)
def test_case1_pipe_raw_producer_repro():
    actual, _ = _launch(_case1_producer_raw_kernel)
    expected = _reference()
    assert torch.equal(actual, expected), _mismatch_diagnostic(actual, expected)


@pytest.mark.xfail(
    strict=True,
    reason="raw consumer uses CTA-global thread ids and ordinary strides in a worker partition",
)
def test_case2_pipe_raw_consumer_repro():
    actual, _ = _launch(_case2_consumer_raw_kernel)
    expected = _reference()
    assert torch.equal(actual, expected), _mismatch_diagnostic(actual, expected)
