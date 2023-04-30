from gql import Client, gql

from ..utils.config import get_config_filepath, read_config_file, update_config_file


def whoami(session: Client):
    query = gql(
        """
        query whoami {
            whoami {
            username
            }
        }
    """
    )
    result = session.execute(query)
    return result


def setup_config(
    username: str,
    token: str,
    host: str = "api.spice.cloud",
    transport: str = "https",
):
    new_config = read_config_file()
    new_host_config = {
        "username": username,
        "token": token,
        "transport": transport,
    }
    if new_config.get(host, None):
        new_config[host] = new_host_config
    else:
        new_config[host] = new_host_config
    update_config_file(new_config=new_config)
    return get_config_filepath()
