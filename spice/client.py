
from spice.auth.actions import Auth
from spice.graphql.sdk import create_session
from spice.hardware.actions import Hardware
from spice.utils.config import read_config_file


class Spice:
    def __init__(self, host: str = "api.spice.cloud") -> None:
        full_config = read_config_file()

        self.config = full_config.get(host)
        if not self.config:
            raise KeyError(f"Host {host} not found in config file.")

        self.session = create_session(host=host)

        self.auth = Auth(self)
        self.hardware = Hardware(self)
