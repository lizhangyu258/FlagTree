from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.cuda.runtime import _resolve_clang


BLOCK = 128
_CUDA_SOURCE = Path(__file__).with_name("tle_pipe_raw_sync.cu")


def _has_hopper_and_raw_clang():
    try:
        if not torch.cuda.is_available():
            return False
        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "cuda":
            return False
        if torch.cuda.get_device_capability()[0] < 9:
            return False
        _resolve_clang()
        return True
    except (AttributeError, RuntimeError, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _has_hopper_and_raw_clang(),
    reason="requires NVIDIA Hopper (sm90+) and TLE raw clang",
)


@dialect(
    name="cuda",
    file=_CUDA_SOURCE,
    extern_func_name="raw_pipe_producer",
    deferred=False,
)
def _raw_pipe_producer(*args, **kwargs):
    ...


@dialect(
    name="cuda",
    file=_CUDA_SOURCE,
    extern_func_name="raw_pipe_consumer",
    deferred=False,
)
def _raw_pipe_consumer(*args, **kwargs):
    ...


@triton.jit
def _raw_producer_kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    stage_buf = tle.gpu.alloc(
        [1, BLOCK_SIZE],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=1, scope="cta", name="raw_producer", x=stage_buf)
    writer = pipe.writer()
    reader = pipe.reader()

    slot = writer.acquire(0)
    alias = tle_raw.call_smem(
        _raw_pipe_producer,
        [slot.x, input_ptr],
        output_indices=[0],
        effects=["write", "read"],
    )
    writer.commit(0)
    reader.wait(0)
    value = tl.load(tle.gpu.local_ptr(alias))
    tl.store(output_ptr + tl.arange(0, BLOCK_SIZE), value)
    reader.release(0)


@triton.jit
def _raw_consumer_kernel(input_ptr, output_ptr, probe_ptr, BLOCK_SIZE: tl.constexpr):
    stage_buf = tle.gpu.alloc(
        [1, BLOCK_SIZE],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=1, scope="cta", name="raw_consumer", x=stage_buf)
    writer = pipe.writer()
    reader = pipe.reader()
    offsets = tl.arange(0, BLOCK_SIZE)

    slot = writer.acquire(0)
    value = tl.load(input_ptr + offsets)
    tl.store(tle.gpu.local_ptr(slot.x), value)
    writer.commit(0)
    wait = reader.wait(0)
    alias = tle_raw.call_smem(
        _raw_pipe_consumer,
        [wait.slot.x, output_ptr],
        output_indices=[0],
        effects=["read", "write"],
    )
    probe = tl.load(tle.gpu.local_ptr(alias))
    tl.store(probe_ptr + offsets, probe)
    reader.release(0)


def test_raw_pipe_producer():
    source = torch.randn(BLOCK, device="cuda", dtype=torch.float32)
    output = torch.full_like(source, float("nan"))
    _raw_producer_kernel[(1, )](source, output, BLOCK_SIZE=BLOCK, num_warps=4)
    torch.testing.assert_close(output, source, atol=0, rtol=0)


def test_raw_pipe_consumer():
    source = torch.randn(BLOCK, device="cuda", dtype=torch.float32)
    output = torch.full_like(source, float("nan"))
    probe = torch.full_like(source, float("nan"))
    _raw_consumer_kernel[(1, )](source, output, probe, BLOCK_SIZE=BLOCK, num_warps=4)
    torch.testing.assert_close(output, source, atol=0, rtol=0)
    torch.testing.assert_close(probe, source, atol=0, rtol=0)
