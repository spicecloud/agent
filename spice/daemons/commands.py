import click

from ..utils.printer import print_result


@click.group()
@click.pass_context
def cli(context):
    """Daemon"""
    pass


@cli.command("install")
@click.pass_context
def install_daemon_command(context):
    """Install the full spice Daemon"""
    spice = context.obj.get("SPICE")
    result = spice.daemons.install()
    print_result(message=result, context=context)


@cli.command("stop")
@click.pass_context
def stop_daemon_command(context):
    """Stop spice Daemon"""
    spice = context.obj.get("SPICE")
    result = spice.daemons.stop()
    message = f"stopping spice daemon"
    if context.obj["JSON"]:
        message = result
    print_result(message=message, context=context)


@cli.command("start")
@click.pass_context
def start_daemon_command(context):
    """Start spice Daemon"""
    spice = context.obj.get("SPICE")
    result = spice.daemons.start()
    message = f"starting spice daemon"
    if context.obj["JSON"]:
        message = result
    print_result(message=message, context=context)


@cli.command("logs")
@click.pass_context
def log_daemon_command(context):
    """Logs from spice Daemon"""
    spice = context.obj.get("SPICE")
    result = spice.daemons.logs()
    print_result(message=result, context=context)
