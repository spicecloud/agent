import json
import logging
import platform

from gql import gql
import pika
import torch
from transformers import pipeline
from transformers.pipelines.base import PipelineException

LOGGER = logging.getLogger(__name__)


class Inference:
    def __init__(self, spice) -> None:
        self.spice = spice
        self.device = self.get_device()

        # logging.basicConfig(level=logging.INFO)
        if not self.spice.DEBUG:
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("pika").setLevel(logging.ERROR)

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
            if device is None and self.spice.DEBUG:
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
            if device is None and self.spice.DEBUG:
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
            return (
                f"PyTorch installed correctly and tensors ran on device: {self.device}"
            )
        except Exception as exception:
            return str(exception)

    def update_run_status(self, run_id: str, status: str, result: str):
        mutation = gql(
            """
            mutation updateRunStatus($runId: String!, $status: String!, $result: String) {
                updateRunStatus(runId: $runId, status: $status, result: $result) {
                    runId
                    status
                }
            }
        """  # noqa
        )
        variables = {"runId": run_id, "status": status, "result": result}
        result = self.spice.session.execute(mutation, variable_values=variables)
        return result

    def run_pipeline(self, model="bert-base-uncased", input="spice.cloud is [MASK]!"):
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

        pipe = pipeline(model=model, device=self.device)
        result = pipe(input)
        return result

    def run_pipeline_callback(self, channel, method, properties, body: str):
        body = json.loads(body.decode("utf-8"))
        model = body["model"]
        if not model:
            raise Exception(f'No model found in message body: {body.decode("utf-8")}')

        new_input = body["input"]
        if not new_input:
            raise Exception(f'No input found in message body: {body.decode("utf-8")}')

        run_id = body["run_id"]

        try:
            result = self.run_pipeline(model=model, input=new_input)
            print(f"run_id: {run_id}. result: {result}")
            self.update_run_status(
                run_id=run_id, status="SUCCESS", result=json.dumps(result)
            )
        except PipelineException as exception:
            message = f"""Input: "{new_input}" threw exception: {exception}"""
            LOGGER.error(message)
            self.update_run_status(
                run_id=run_id, status="ERROR", result=json.dumps(message)
            )

    def worker(self):
        credentials = pika.PlainCredentials(
            self.spice.host_config["fingerprint"],
            self.spice.host_config["rabbitmq_password"],
        )
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.spice.host_config["rabbitmq_host"],
                port=self.spice.host_config["rabbitmq_port"],
                virtual_host="inference",
                credentials=credentials,
            )
        )
        channel = connection.channel()
        print(" [*] Waiting for messages. To exit press CTRL+C")
        while True:
            channel.basic_consume(
                queue="default",
                on_message_callback=self.run_pipeline_callback,
                auto_ack=True,
            )
            channel.start_consuming()
