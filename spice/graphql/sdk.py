from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from ..utils.config import read_config_file


def create_session(
    host: str = "api.spice.cloud",
    fetch_schema_from_transport: bool = False,
):
    existing_config = read_config_file()
    token = existing_config.get(host).get("token")
    transport = existing_config.get(host).get("transport")
    url = f"{transport}://{host}/"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    transport = AIOHTTPTransport(url=url, headers=headers)
    session = Client(
        transport=transport,
        fetch_schema_from_transport=fetch_schema_from_transport,
    )
    return session


def login_mutation(session: Client, username: str, password: str):
    mutation = gql(
        """
        mutation login($username: String!, $password: String!) {
            login (username: $username, password: $password) {
                username
            }
        }
    """
    )
    params = {"username": username, "password": password}
    result = session.execute(mutation, variable_values=params)
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


def whoami_query(session: Client):
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
