


class Job:
    def __init__(self, spice) -> None:
        self.spice = spice

    def verify_torch(self):
        import torch

        try:
            torch.rand(5, 3)
            return "PyTorch installed correctly."
        except Exception as exception:
            return str(exception)
