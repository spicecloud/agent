import os
import sys

import click

from spice.client import Spice

from .auth.commands import config_command, whoami_command
from .hardware.commands import cli as hardware_cli
from .inference.commands import cli as inference_cli


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
    if "config" not in sys.argv:
        context.obj["SPICE"] = Spice(host=host, DEBUG=debug)


cli.add_command(config_command, "config")
cli.add_command(whoami_command, "whoami")
cli.add_command(hardware_cli, "hardware")
cli.add_command(inference_cli, "inference")


if __name__ == "__main__":
    cli(obj={})
