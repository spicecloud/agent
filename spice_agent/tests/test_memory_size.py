# tests/test_memory_size.py

import pytest
from spice_agent.utils.memory_size import MemorySize


def test_invalid_value_initialization():
    # Unit should never be None in a value-based initialization
    with pytest.raises(ValueError):
        MemorySize(512, None)

    # An unsupported unit type should never be passed int
    with pytest.raises(ValueError):
        MemorySize(512, "invalid_unit")


def test_invalid_string_initialization():
    # There should be a space between the value and the unit
    with pytest.raises(ValueError):
        MemorySize("1.5TB")

    # No numeric value
    with pytest.raises(ValueError):
        MemorySize("invalid_unit MB")

    with pytest.raises(ValueError):
        MemorySize("3.14 xyz")  # Invalid unit


def test_to_bytes_conversion():
    # Check same unit
    size = MemorySize(512, "B")
    assert size.to_bytes() == 512

    size = MemorySize("512 B")
    assert size.to_bytes() == 512

    # Check going up in scale
    size = MemorySize(1024, "b")
    assert size.to_bytes() == 128

    size = MemorySize("1024 b")
    assert size.to_bytes() == 128

    # Check going down in scale
    size = MemorySize(1.03, "GB")
    assert size.to_bytes() == 1.03 * 1024**3

    size = MemorySize("1.03 GB")
    assert size.to_bytes() == 1.03 * 1024**3

    # Check going from binary to decimal
    size = MemorySize(2.123, "PiB")
    assert size.to_bytes() == 2.123 * 2**50

    size = MemorySize("2.123 PiB")
    assert size.to_bytes() == 2.123 * 2**50


def test_convert_to():
    size = MemorySize(512, "MiB")

    # Check same unit
    size.convert_to("MiB")
    assert str(size) == "512.000 MiB"

    # Convert down
    size.convert_to("KiB")
    assert str(size) == "524288.000 KiB"

    # Convert up
    size.convert_to("GiB")
    assert str(size) == "0.500 GiB"

    # Convert from binary to decimal
    size.convert_to("B")
    assert str(size) == "536870912.000 B"

    # Test invalid unit
    with pytest.raises(ValueError):
        size.convert_to("invalid")


def test_comparison_equal():
    size1 = MemorySize(1024, "KB")
    size2 = MemorySize(1, "MB")
    assert size1 == size2


def test_comparison_not_equal():
    size1 = MemorySize(512, "B")
    size2 = MemorySize(1, "KB")
    assert size1 != size2


def test_comparison_type_error():
    size = MemorySize(512, "B")
    with pytest.raises(TypeError):
        result = size == "512 B"
        assert result is False
