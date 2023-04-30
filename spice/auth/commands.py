import json
import os
import webbrowser

import click

from .actions import setup_config, whoami


@click.group()
@click.pass_context
def cli(context):
    """Authentication"""
    pass


@cli.command("whoami")
@click.pass_context
def whoami_command(context):
    """Whoami"""
    session = context.obj.get("SESSION")
    message = whoami(session=session)
    if context.obj["JSON"]:
        message = json.dumps(message)
    click.echo(message)


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
    "--host",
    required=False,
    default=lambda: os.environ.get("SPICE_HOST", "api.spice.cloud"),
    show_default="api.spice.cloud",
    help="Environment of Spice API",
)
@click.option(
    "--transport",
    required=False,
    default="https",
    show_default="https",
    help="Transport protocol for Spice API",
)
def config_command(context, username: str, host: str, transport: str):
    """Configure the CLI"""
    token = os.environ.get("SPICE_TOKEN", "")
    if not token and context.obj.get("YES"):
        raise click.ClickException(
            "SPICE_TOKEN environment variable is required for non-interactive mode"
        )
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

    config_path = setup_config(
        username=username, token=token, host=host, transport=transport
    )
    message = (
        f"Config created at '{config_path}' for user '{username}' on host '{host}'"
    )

    if context.obj["JSON"]:
        message = json.dumps({"result": message})
    click.secho(message, fg="green")
