from gql import Client
from gql.transport.aiohttp import AIOHTTPTransport

from spice.__version__ import __version__

from ..utils.config import read_config_file


def create_session(
    host: str = "api.spice.cloud",
    fingerprint: str = None,
    fetch_schema_from_transport: bool = False,
):
    print("reading @ create_session")
    existing_config = read_config_file()
    host_config = existing_config.get(host)
    if not host_config:
        raise KeyError(f"Host {host} not found in config file.")

    token = host_config.get("token")
    transport = host_config.get("transport")
    url = f"{transport}://{host}/"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-spice-agent": f"spice@{__version__}",
    }
    if token:
        headers["Authorization"] = f"Token {token}"

    if fingerprint:
        headers["x-spice-fingerprint"] = fingerprint
    transport = AIOHTTPTransport(url=url, headers=headers)
    session = Client(
        transport=transport,
        fetch_schema_from_transport=fetch_schema_from_transport,
    )
    return session
