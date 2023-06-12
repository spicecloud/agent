from gql import gql

from ..utils.config import read_config_file, update_config_file


class Auth:
    def __init__(self, spice) -> None:
        self.spice = spice

    def whoami(self):
        query = gql(
            """
            query whoami {
                whoami {
                    username
                }
            }
        """
        )
        result = self.spice.session.execute(query)
        return result

    def setup_config(
        self,
        username: str,
        token: str,
        host: str = "api.spice.cloud",
        transport: str = "https",
    ):
        full_config = read_config_file()
        new_host_config = {
            "username": username,
            "token": token,
            "transport": transport,
        }

        if existing_host_config := full_config.get(host, None):
            full_config[host] = {**existing_host_config, **new_host_config}
        else:
            full_config[host] = new_host_config
        update_config_file(new_config=full_config)
