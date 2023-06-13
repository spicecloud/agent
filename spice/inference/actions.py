import json
import logging
import os

from gql import gql

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch  # noqa
import transformers  # noqa
from transformers import pipeline, set_seed  # noqa
from transformers.pipelines.base import PipelineException  # noqa

LOGGER = logging.getLogger(__name__)


class Inference:
    def __init__(self, spice) -> None:
        self.spice = spice
        self.device = self.spice.get_device()

        # logging.basicConfig(level=logging.INFO)
        if self.spice.DEBUG:
            transformers.logging.set_verbosity_debug()
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("pika").setLevel(logging.ERROR)

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
            return (
                f"PyTorch installed correctly and tensors ran on device: {self.device}"
            )
        except Exception as exception:
            return str(exception)

    def update_run_status(self, run_id: str, status: str, result: str):
        mutation = gql(
            """
            mutation updateRunStatus($input: UpdateRunStatusInput!) {
                updateRunStatus(input: $input) {
                    ... on CurrentRunStatus {
                        runId
                        status
                    }
                }
            }
        """  # noqa
        )
        input = {"runId": run_id, "status": status, "result": result}
        variables = {"input": input}
        result = self.spice.session.execute(mutation, variable_values=variables)
        return result

    def run_bert_base_uncased(self, run_id: str, input="spice.cloud is [MASK]!"):
        pipe = pipeline(model="bert-base-uncased", device=self.device)
        result = pipe(input)
        return result

    def run_gpt2(self, run_id: str, input="Hello, I'm a language model,"):
        generator = pipeline("text-generation", model="gpt2", device=self.device)
        set_seed(42)
        result = generator(input, max_length=30, num_return_sequences=1)
        return result

    def run_pipeline(
        self, run_id: str, model="bert-base-uncased", new_input="spice.cloud is [MASK]!"
    ):  # noqa
        LOGGER.info(f""" [*] Run ID: {run_id}. Model: {model}. Input: {new_input}""")
        try:
            result = None
            if model == "bert-base-uncased":
                result = self.run_bert_base_uncased(run_id=run_id, input=new_input)
            elif model == "gpt2":
                result = self.run_gpt2(run_id=run_id, input=new_input)
            else:
                raise Exception(
                    f"Unsupported model {model}. Please email support for addition."
                )
            LOGGER.info(f""" [*] SUCCESS. Result: " {result}""")
            self.update_run_status(
                run_id=run_id, status="SUCCESS", result=json.dumps(result)
            )
        except PipelineException as exception:
            message = f"""Input: "{input}" threw exception: {exception}"""
            LOGGER.error(message)
            self.update_run_status(
                run_id=run_id, status="ERROR", result=json.dumps(message)
            )

    # def run_pipeline(self, task="", model="bert-base-uncased", input="spice.cloud is [MASK]!"):  # noqa
    #     # # Load pre-trained model tokenizer (vocabulary)
    #     # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #     # # Load pre-trained model (weights)
    #     # model = BertModel.from_pretrained("bert-base-uncased")
    #     # model.eval()  # Set model to evaluation mode
    #     # text = "Replace this with your text input"
    #     # encoded_input = tokenizer(text, return_tensors="pt")
    #     # with torch.no_grad():  # Disable gradient calculations
    #     #     output = model(**encoded_input)  # Forward pass
    #     # # 'output' now contains the model's output
    #     # print(output)
    #     # from transformers import BertTokenizer, BertModel
    #     # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     # model = BertModel.from_pretrained("bert-base-uncased")
    #     # text = "Replace me by any text you'd like."
    #     # encoded_input = tokenizer(text, return_tensors='pt')
    #     # output = model(**encoded_input)

    #     pipe = pipeline(model=model, device=self.device)
    #     result = pipe(input)
    #     return result
