import os

import click

from spice.graphql.sdk import create_session

from .auth.commands import config_command, whoami_command
from .hardware.commands import cli as hardware_cli
from .utils.config import read_config_file


@click.group()
@click.option(
    "--host",
    required=False,
    default=lambda: os.environ.get("SPICE_HOST", "api.spice.cloud"),
    show_default="api.spice.cloud",
    help="Environment of Spice API",
)
@click.option(
    "--yes", is_flag=True, show_default=True, default=False, help="Skip interactions."
)
@click.option(
    "--debug", is_flag=True, show_default=True, default=False, help="Enable debug mode."
)
@click.option(
    "--json",
    is_flag=True,
    show_default=True,
    default=False,
    help="Turn all outputs into JSON.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity of output levels.",
)
@click.pass_context
def cli(context, host, yes, debug, json, verbose):
    """spice agent - a CLI tool for https://spice.cloud"""
    context.ensure_object(dict)
    context.obj["YES"] = yes
    context.obj["DEBUG"] = debug
    context.obj["JSON"] = json
    context.obj["VERBOSE"] = verbose

    existing_config = read_config_file()
    context.obj["CONFIG"] = existing_config
    if host:
        context.obj["HOST"] = host
    context.obj["SESSION"] = create_session(host=host)


cli.add_command(config_command, "config")
cli.add_command(whoami_command, "whoami")
cli.add_command(hardware_cli, "hardware")


if __name__ == "__main__":
    cli(obj={})
