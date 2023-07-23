import os
import webbrowser

import click

from spice_agent.auth.actions import Auth
from spice_agent.client import Spice

from ..utils.config import SPICE_HOSTS_FILEPATH
from ..utils.printer import print_result


@click.group()
@click.pass_context
def cli(context):
    """Authentication"""
    pass


@cli.command("whoami")
@click.pass_context
def whoami_command(context):
    """Whoami"""
    spice = context.obj.get("SPICE")
    result = spice.auth.whoami()
    message = f"Logged in as {result['whoami']['username']}"
    if context.obj["JSON"]:
        message = result
    print_result(message=message, context=context)


@cli.command("config")
@click.pass_context
@click.option(
    "--username",
    prompt=True,
    default=lambda: os.environ.get("SPICE_USER", ""),
    show_default="current user",
    help="Username for Spice API",
)
@click.option(
    "--transport",
    required=False,
    default="https",
    show_default="https",
    help="Transport protocol for Spice API",
)
@click.option(
    "--register",
    is_flag=True,
    show_default=True,
    default=False,
    help="Auto Register Hardware with Spice Cloud",
)
def config_command(context, username: str, transport: str, register: bool):
    """Configure the CLI"""
    token = os.environ.get("SPICE_TOKEN", "")
    if not token and context.obj.get("YES"):
        raise click.ClickException(
            "SPICE_TOKEN environment variable is required for non-interactive mode"
        )
    host = context.obj.get("HOST", "api.spice.cloud")
    while not token:
        settings_url = f"{transport}://{host.strip('api.')}/settings"
        if "localhost" in host:
            settings_url = "http://localhost:3000/settings"
        click.secho(
            f"You can find or create a Spice API token at {settings_url}",
            fg="yellow",
        )

        webbrowser.open_new_tab(settings_url)
        token = click.prompt(
            "Please enter your Spice API token", hide_input=True, type=str
        )

    auth = Auth(spice=None)
    auth.setup_config(username=username, token=token, host=host, transport=transport)
    message = f"Config created at '{SPICE_HOSTS_FILEPATH}' for user '{username}' on host '{host}'"  # noqa
    print_result(message=message, context=context, fg="green")

    if register or click.confirm(
        "Do you want to register this machine with spice.cloud?"
    ):
        spice = Spice(host=host)
        result = spice.hardware.register()

        if result.get("registerHardware"):
            print_result(
                message="Hardware registered successfully", context=context, fg="green"
            )

    if register or click.confirm(
        "Do you want to install the agent as a background process?"
    ):
        spice = Spice(host=host)
        spice.daemons.install()
        print_result(
            message="Daemon installed. View its logs with `spice daemon logs`",
            context=context,
            fg="green",
        )
