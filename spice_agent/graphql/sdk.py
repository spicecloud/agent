from typing import Dict

from gql import Client
from gql.transport.aiohttp import AIOHTTPTransport
import requests

from spice_agent.__version__ import __version__


def create_session(
    host_config: Dict,
    host: str = "api.spice.cloud",
    fetch_schema_from_transport: bool = False,
):
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

    if fingerprint := host_config.get("fingerprint", None):
        headers["x-spice-fingerprint"] = fingerprint
    transport = AIOHTTPTransport(url=url, headers=headers)
    session = Client(
        transport=transport,
        fetch_schema_from_transport=fetch_schema_from_transport,
    )
    return session


def create_requests_session(
    host_config: Dict,
):
    token = host_config.get("token")

    headers = {
        # "Content-Type": "multipart/form-data", # DO NOT ADD THIS MIME TYPE IT BREAKS
        "Accept": "application/json",
        "x-spice-agent": f"spice@{__version__}",
    }
    if token:
        headers["Authorization"] = f"Token {token}"

    if fingerprint := host_config.get("fingerprint", None):
        headers["x-spice-fingerprint"] = fingerprint

    session = requests.Session()
    session.headers.update(headers)
    return session
