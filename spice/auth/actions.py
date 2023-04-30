from ..graphql.sdk import login_mutation, sdk_client, whoami_query
from ..utils.config import get_config_filepath, read_config_file, update_config_file

DEFAULT_HOST = "localhost:8000"


def login(username: str, password: str):
    return


def whoami():
    existing_config = read_config_file()
    token = existing_config.get(DEFAULT_HOST).get("token")
    transport = existing_config.get(DEFAULT_HOST).get("transport")
    host = DEFAULT_HOST
    url = f"{transport}://{host}/"
    session = sdk_client(url=url, token=token)
    result = whoami_query(session=session)
    return result


def get_token(
    username: str,
    password: str,
    host: str = "api.spice.cloud",
    transport: str = "https",
):
    token = ""
    url = f"{transport}://{host}/"
    session = sdk_client(url=url)
    login_mutation(session=session, username=username, password=password)
    whoami_query(session=session)
    # create_auth_token(session=session)
    return token


def setup_config(
    username: str,
    password: str,
    host: str = "api.spice.cloud",
    transport: str = "https",
):
    token = get_token(
        username=username, password=password, host=host, transport=transport
    )
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
