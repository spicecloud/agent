import json

import click

from ..utils.printer import print_result


@click.group()
@click.pass_context
def cli(context):
    """Hardware"""
    pass


@cli.command("systeminfo")
@click.pass_context
def systeminfo_command(context):
    """Get System Info"""
    spice = context.obj.get("SPICE")
    message = spice.hardware.get_system_info()
    print_result(message=message, context=context)


@cli.command("register")
@click.pass_context
def register_command(context):
    """Register Hardware with Spice"""
    spice = context.obj.get("SPICE")
    message = spice.hardware.register()
    print_result(message=message, context=context)
