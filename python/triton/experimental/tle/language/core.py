# flagtree tle
import triton.language.core as tl

def _tle_pick_sum_dtype(in_dtype, dtype):
    if dtype is not None:
        return dtype
    if in_dtype.is_int_signed():
        return tl.int32 if in_dtype.int_bitwidth < 32 else None
    if in_dtype.is_int_unsigned():
        return tl.uint32 if in_dtype.int_bitwidth < 32 else None
    return None

# -----------------------
# Non-Atomic Memory Operations
# -----------------------


@tl.builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, is_async=False, _semantic=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:

        (1) If `pointer` is a single element pointer, a scalar is be loaded.  In
            this case:

            - `mask` and `other` must also be scalars,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional tensor is loaded.  In this case:

            - `mask` and `other` are implicitly broadcast to `pointer.shape`,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a
            tensor is loaded.  In this case:

            - `mask` and `other` must be `None`, and
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access.

    :param pointer: Pointer to the data to be loaded
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`
        (must be `None` with block pointers)
    :type mask: Block of `triton.int1`, optional
    :param other: if `mask[idx]` is false, return `other[idx]`
    :type other: Block, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param padding_option: should be one of {"", "zero", "nan"}, the padding value to use while out of bounds. "" means an undefined value.
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", ".ca", ".cg", ".cv"}, where ".ca" stands for
        cache at all levels, ".cg" stands for cache at global level (cache in L2 and below, not L1),
        and ".cv" means don’t cache and fetch again. see
        `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    """
    x = tl.load(pointer, mask=mask, other=other, boundary_check=boundary_check, padding_option=padding_option,
                cache_modifier=cache_modifier, eviction_policy=eviction_policy, volatile=volatile, _semantic=_semantic)
    x.handle.set_attr("tt.load.async", _semantic.builder.get_bool_attr(is_async))
    return x


@tl.builtin
def cumsum(input, axis=0, reverse=False, dtype: tl.constexpr = None, _semantic=None, _generator=None):
    """
    Compute exclusive cumulative sum and total sum along :code:`axis`.

    Returns a tuple :code:`(exclusive_sum, total_sum)` where:
    - :code:`exclusive_sum[i] = sum(input[:i])` (or reverse-exclusive when ``reverse=True``)
    - :code:`total_sum = sum(input)`
    """
    axis = tl._unwrap_if_constexpr(axis)
    reverse = tl._unwrap_if_constexpr(reverse)
    dtype = tl._unwrap_if_constexpr(dtype)
    input = tl._promote_bfloat16_to_float32(input, _semantic=_semantic)
    out_dtype: tl.constexpr = _tle_pick_sum_dtype(input.dtype, dtype)
    if out_dtype is not None:
        input = input.to(out_dtype, _semantic=_semantic)

    if not isinstance(input, tl.tensor):
        input = _semantic.to_tensor(input)

    input_ty = input.type
    if not input_ty.is_block():
        zero = tl.full((), 0, input_ty, _semantic=_semantic)
        return zero, input

    exclusive_ty = input_ty
    total_ty = input_ty.scalar
    exclusive_ir = exclusive_ty.to_ir(_semantic.builder)
    total_ir = total_ty.to_ir(_semantic.builder)

    cumsum_op = _semantic.builder.create_exclusive_cumsum(
        exclusive_ir,
        total_ir,
        input.handle,
        int(axis),
        bool(reverse),
    )
    exclusive_sum = tl.tensor(cumsum_op.get_result(0), exclusive_ty)
    total_sum = tl.tensor(cumsum_op.get_result(1), total_ty)
    return exclusive_sum, total_sum
