from ..graphql.sdk import sdk_client, whoami_query
from ..utils.config import get_config_filepath, read_config_file, update_config_file

DEFAULT_HOST = "localhost:8000"


def whoami():
    existing_config = read_config_file()
    token = existing_config.get(DEFAULT_HOST).get("token")
    transport = existing_config.get(DEFAULT_HOST).get("transport")
    host = DEFAULT_HOST
    url = f"{transport}://{host}/"
    session = sdk_client(url=url, token=token)
    result = whoami_query(session=session)
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
