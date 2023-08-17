class MemorySize:
    CONVERSION_FACTORS = {
        "b": 1 / 8,
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
        "KiB": 2**10,
        "MiB": 2**20,
        "GiB": 2**30,
        "TiB": 2**40,
        "PiB": 2**50,
    }

    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

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
        return self.value * self.CONVERSION_FACTORS.get(self.unit)

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

    @classmethod
    def from_string(cls, size_string):
        """
        Creates a MemorySize instance from a size string.
        """
        parts = size_string.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid size string format: {size_string}")

        value_string, unit = parts
        try:
            value = float(value_string)
        except ValueError:
            raise ValueError(f"Invalid size value: {value_string}")

        return cls(value, unit)
