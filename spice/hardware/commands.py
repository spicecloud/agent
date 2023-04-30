import json

import click

from .actions import register


@click.group()
@click.pass_context
def cli(context):
    """Hardware"""
    pass


@cli.command("register")
@click.pass_context
def register_command(context):
    """Register"""
    session = context.obj.get("SESSION")
    message = register(session=session)
    if context.obj["JSON"]:
        message = json.dumps(message, indent=4)
    click.echo(message)
