import asyncio

from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

GRAPHQL_URL = "http://localhost:8000/"


async def main():
    transport = AIOHTTPTransport(url=GRAPHQL_URL)
    async with Client(
        transport=transport,
        fetch_schema_from_transport=True,
    ) as session:
        query = gql(
            """
            mutation login($username: String!, $password: String!) {
                login (username: $username, password: $password) {
                    username
                }
            }
        """
        )

        params = {"username": "", "password": ""}

        result = await session.execute(query, variable_values=params)
        print(result)

        query = gql(
            """
            query whoami {
              whoami {
                username
              }
            }
        """
        )

        result = await session.execute(query)
        print(result)


asyncio.run(main())
