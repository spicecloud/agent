from spice.auth.actions import Auth
from spice.graphql.sdk import create_session
from spice.hardware.actions import Hardware
from spice.job.actions import Job
from spice.utils.config import read_config_file


class Spice:
    def __init__(self, host: str = "api.spice.cloud") -> None:
        self.host = host
        self.full_config = read_config_file()
        self.host_config = self.full_config.get(self.host)

        if not self.host_config:
            raise KeyError(f"Host {self.host} not found in config file.")

        self.session = create_session(
            host=self.host, fingerprint=self.host_config.get("fingerprint", None)
        )

        self.auth = Auth(self)
        self.hardware = Hardware(self)
        self.job = Job(self)
