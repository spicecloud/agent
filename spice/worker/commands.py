import click

from ..utils.printer import print_result


@click.group()
@click.pass_context
def cli(context):
    """Worker"""
    pass


@cli.command("start")
@click.pass_context
def start(context):
    """Run as a worker that picks up tasks from the spice.cloud queue."""
    spice = context.obj.get("SPICE")
    message = spice.worker.start()
    print_result(message=message, context=context)
