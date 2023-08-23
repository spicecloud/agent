from typing import Tuple


class MemorySize:
    CONVERSION_FACTORS: dict[str, float] = {
        "b": 1 / 8,
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "PB": 1000**5,
        "KiB": 2**10,
        "MiB": 2**20,
        "GiB": 2**30,
        "TiB": 2**40,
        "PiB": 2**50,
    }

    def __init__(self, value_or_string: int | float | str, unit: str | None = None):
        """
        Initializes MemorySize in two ways:
            size = MemorySize("3.14 GB") - value_or_string is str and unit is None
            size = MemorySize(42, "MB") - value_or_string is int and unit is str
            size = MemorySize(2.718, "PB") - value_or_string is float and unit is str
        """
        if isinstance(value_or_string, str):
            parts = value_or_string.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid size string format: {value_or_string}")

            value_string, unit = parts
            try:
                self.value = float(value_string)
            except ValueError:
                raise ValueError(f"Invalid size value: {value_string}")
            self.unit = unit
        elif isinstance(value_or_string, int | float) and isinstance(unit, str):
            self.value = float(value_or_string)
            self.unit = unit
        else:
            message = f"Invalid arguments provided: {value_or_string}:{type(value_or_string)} and {unit}:{type(unit)}"  # noqa
            raise ValueError(message)

        if not self.CONVERSION_FACTORS.get(self.unit):
            message = f"Invalid unit: {self.unit}"
            raise ValueError(message)

    def __str__(self):
        return f"{self.value:.3f} {self.unit}"

    def __eq__(self, other):
        if not isinstance(other, MemorySize):
            raise TypeError(f"Comparison with unsupported type: {type(other)}")
        return self.to_bytes() == other.to_bytes()

    def to_bytes(self):
        """
        Converts value to bytes (B)
        """
        return self.value * self.CONVERSION_FACTORS.get(self.unit, 1)

    def convert_to(self, target_unit):
        """
        Converts the stored value to a different unit and updates the instance.
        """
        if target_unit not in self.CONVERSION_FACTORS:
            raise ValueError(f"Invalid unit {target_unit}")

        factor = (
            self.CONVERSION_FACTORS[self.unit] / self.CONVERSION_FACTORS[target_unit]
        )
        self.value *= factor
        self.unit = target_unit

    def get_human_readable(self) -> Tuple[float, str]:
        """
        Converts the stored value to a human-readable format
        """
        HUMAN_READABLE_UNITS: list[str] = ["PB", "TB", "GB", "MB", "KB", "B"]

        value_in_bytes = self.to_bytes()
        for unit in HUMAN_READABLE_UNITS:
            value_in_unit = value_in_bytes * (1.000 / self.CONVERSION_FACTORS[unit])
            if value_in_unit >= 1.0:
                return (value_in_unit, unit)
        return (value_in_bytes, "B")
