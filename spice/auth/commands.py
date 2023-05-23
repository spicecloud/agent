import os
import webbrowser

import click

from spice.auth.actions import Auth

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
def config_command(context, username: str, transport: str):
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
    message = f"Config created at '{SPICE_HOSTS_FILEPATH}' for user '{username}' on \
        host '{host}'"
    print_result(message=message, context=context, fg="green")
