import sentry_sdk
from sentry_sdk import set_user

import spice

from spice.auth.actions import Auth
from spice.graphql.sdk import create_session
from spice.hardware.actions import Hardware
from spice.inference.actions import Inference
from spice.utils.config import read_config_file


class Spice:
    def __init__(self, host: str = "api.spice.cloud", DEBUG: bool = False) -> None:
        self.host = host
        self.DEBUG = DEBUG

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
            release=f"spice@{spice.__version__}",
        )

        self.full_config = read_config_file()
        self.host_config = self.full_config.get(self.host)

        if not self.host_config:
            raise KeyError(f"Host {self.host} not found in config file.")
        else:
            set_user({"username": self.host_config["username"]})

        self.session = create_session(
            host=self.host, fingerprint=self.host_config.get("fingerprint", None)
        )

        self.auth = Auth(self)
        self.hardware = Hardware(self)
        self.inference = Inference(self)
