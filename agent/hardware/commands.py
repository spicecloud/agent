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
    result = spice.hardware.register()
    if result.get("registerHardware"):
        message = "Hardware registered successfully"
    print_result(message=message, context=context, fg="green")


@cli.command("checkin")
@click.pass_context
def checkin_command(context):
    """Checkin Hardware with Spice"""
    spice = context.obj.get("SPICE")
    result = spice.hardware.check_in_http()
    if result.get("checkIn"):
        message = "Hardware checked in successfully"
    print_result(message=message, context=context, fg="green")
