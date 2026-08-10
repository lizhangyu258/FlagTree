import pytest
import triton.language as tl

from triton.experimental.tle.language.raw.core import _normalize_effects


def test_normalize_effects():
    args = [object(), object(), object(), object()]
    assert _normalize_effects(["none", "read", "write", "read_write"], args) == [0, 1, 2, 3]
    assert _normalize_effects(tl.constexpr(("read_write", "none")), args[:2]) == [3, 0]
    assert _normalize_effects(None, args) is None


def test_effects_length_validation():
    with pytest.raises(ValueError, match="one entry per argument"):
        _normalize_effects(["read"], [object(), object()])


def test_effects_enum_validation():
    with pytest.raises(ValueError, match="invalid tle_raw.call effect"):
        _normalize_effects(["readonly"], [object()])


def test_effects_type_validation():
    with pytest.raises(TypeError, match="sequence or None"):
        _normalize_effects("read", [object()])
