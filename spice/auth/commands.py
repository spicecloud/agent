import json
import os

import click

from .actions import login, setup_config, whoami


@click.group()
@click.pass_context
def cli(context):
    """Authentication"""
    pass


@cli.command("login")
@click.pass_context
def login_command(context):
    """Login"""
    message = login()
    if context.obj["JSON"]:
        message = json.dumps({"result": message})
    click.echo(message)


@cli.command("whoami")
@click.pass_context
def whoami_command(context):
    """Whoami"""
    message = whoami()
    if context.obj["JSON"]:
        message = json.dumps({"result": message})
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
@click.password_option(
    help="Password for Spice API",
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
def config_command(context, username: str, password: str, host: str, transport: str):
    """Configure the CLI"""
    config_path = setup_config(
        username=username, password=password, host=host, transport=transport
    )
    message = (
        f"Config created at '{config_path}' for user '{username}' on host '{host}'"
    )
    if context.obj["JSON"]:
        message = json.dumps({"result": message})
    click.secho(message, fg="green")
