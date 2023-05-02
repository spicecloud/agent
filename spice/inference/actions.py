import logging
import platform

import torch


class Inference:
    def __init__(self, spice) -> None:
        self.spice = spice
        self.device = self.get_device()

        if not self.spice.DEBUG:
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    def get_device(self):
        """
        First check if mps is available as a device
        Then check for a CUDA device
        Finally, fall back to CPU
        """
        device = None
        os_family = platform.system()

        # mps device enables high-performance training on GPU for macOS
        # devices with Metal programming framework
        # https://pytorch.org/docs/master/notes/mps.html
        if os_family == "Darwin" and torch.backends.mps.is_available():
            device = torch.device("mps")
            if self.spice.DEBUG:
                print("Using MPS device.")
        else:
            if self.spice.DEBUG:
                # in debug mode why is it not available
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not built with MPS enabled."  # noqa
                    )
                else:
                    print(
                        "MPS not available because the current macOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."  # noqa
                    )

        if device is None and torch.cuda.is_available():
            device = torch.device("cuda:0")
            if self.spice.DEBUG:
                print("Using CUDA device.")
        else:
            if self.spice.DEBUG:
                # in debug mode why is it not available
                if not torch.backends.cuda.is_built():
                    print(
                        "CUDA not available because the current PyTorch install was not built with CUDA enabled."  # noqa
                    )
                else:
                    print(
                        "CUDA not available because the current you do not have an CUDA-enabled device on this machine."  # noqa
                    )

        if device is None:
            # fallback to CPU
            device = torch.device("cpu")
            if self.spice.DEBUG:
                print("Using cpu.")

        return device

    def verify_torch(self):
        try:
            # Create a Tensor directly on the mps device
            example_tensor = torch.rand(5, 3, device=self.device)

            # Any operation happens on the GPU
            example_tensor * 2

            # Move your model to mps just like any other device
            # model = YourFavoriteNet()
            # model.to(mps_device)

            # Now every call runs on the GPU
            # pred = model(x)
            return "PyTorch installed correctly."
        except Exception as exception:
            return str(exception)

    def run_inference(self, model="bert-base-uncased", input="spice.cloud is [MASK]!"):
        # # Load pre-trained model tokenizer (vocabulary)
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # # Load pre-trained model (weights)
        # model = BertModel.from_pretrained("bert-base-uncased")
        # model.eval()  # Set model to evaluation mode
        # text = "Replace this with your text input"
        # encoded_input = tokenizer(text, return_tensors="pt")
        # with torch.no_grad():  # Disable gradient calculations
        #     output = model(**encoded_input)  # Forward pass
        # # 'output' now contains the model's output
        # print(output)
        # from transformers import BertTokenizer, BertModel
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model = BertModel.from_pretrained("bert-base-uncased")
        # text = "Replace me by any text you'd like."
        # encoded_input = tokenizer(text, return_tensors='pt')
        # output = model(**encoded_input)
        from transformers import pipeline

        pipe = pipeline(model=model, device=self.device)
        result = pipe(input)
        return result
