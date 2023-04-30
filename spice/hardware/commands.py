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
    message = register()
    if context.obj["JSON"]:
        message = json.dumps({"result": message})
    click.echo(message)
