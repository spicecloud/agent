import json

import click

from .actions import login


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
    message = "login"
    if context.obj["JSON"]:
        message = json.dumps({"result": message})
    click.echo(message)
