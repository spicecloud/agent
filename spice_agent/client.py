import logging
import os
import platform
import sys

import sentry_sdk
from sentry_sdk import set_user
from torch.mps import empty_cache as mps_empty_cache

from spice_agent.__version__ import __version__
from spice_agent.auth.actions import Auth
from spice_agent.daemons.actions import Daemons
from spice_agent.graphql.sdk import create_session
from spice_agent.hardware.actions import Hardware
from spice_agent.inference.actions import Inference
from spice_agent.training.actions import Training
from spice_agent.uploader.actions import Uploader
from spice_agent.utils.config import read_config_file
from spice_agent.utils.version import update_if_outdated
from spice_agent.worker.actions import Worker

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch  # noqa


class Spice:
    def __init__(self, host: str = "api.spice.cloud", DEBUG: bool = False) -> None:
        update_if_outdated()
        self.host = host
        self.DEBUG = DEBUG

        logger = logging.getLogger("spice_agent")
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",  # noqa
            datefmt="%a, %d %b %Y %H:%M:%S",
        )
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        environment = "production"
        if "localhost" in host:
            environment = "development"
        if "staging" in host:
            environment = "staging"

        # https://docs.sentry.io/product/sentry-basics/dsn-explainer/
        # It's okay to send the user's the Sentry DSN!
        # That's how we get metrics / errors from their usage.
        SENTRY_DSN = "https://1ee4a12126f0421bbe382d9227e46c4a@o4505155992223744.ingest.sentry.io/4505156109139968"  # noqa
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            environment=environment,
            debug=DEBUG,
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
            attach_stacktrace=True,
            send_default_pii=True,
            release=f"spice@{__version__}",
        )

        self.create_session()

        self.worker = Worker(self)
        self.auth = Auth(self)
        self.hardware = Hardware(self)
        self.inference = Inference(self)
        self.training = Training(self)
        self.uploader = Uploader(self)
        self.daemons = Daemons(self)

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
        if os_family == "Darwin" and torch.backends.mps.is_available():  # type: ignore
            device = torch.device("mps")
            mps_empty_cache()
            if self.DEBUG:
                print("Using MPS device.")
        else:
            if device is None and self.DEBUG:
                # in debug mode why is it not available
                if not torch.backends.mps.is_built():  # type: ignore
                    print(
                        "MPS not available because the current PyTorch install was not built with MPS enabled."  # noqa
                    )
                else:
                    print(
                        "MPS not available because the current macOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."  # noqa
                    )

        if device is None and torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.empty_cache()
            if self.DEBUG:
                print("Using CUDA device.")
        else:
            if device is None and self.DEBUG:
                # in debug mode why is it not available
                if not torch.backends.cuda.is_built():  # type: ignore
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
            if self.DEBUG:
                print("Using cpu.")

        return device

    def create_session(self):
        self.full_config = read_config_file()
        self.host_config = self.full_config.get(self.host)

        if not self.host_config:
            raise KeyError(f"Host {self.host} not found in config file.")
        else:
            set_user({"username": self.host_config["username"]})

        self.session = create_session(host=self.host, host_config=self.host_config)
