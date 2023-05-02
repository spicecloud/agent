import click

from ..utils.printer import print_result


@click.group()
@click.pass_context
def cli(context):
    """Job"""
    pass


@cli.command("verify-torch")
@click.pass_context
def verify_torch_command(context):
    """Get System Info"""
    spice = context.obj.get("SPICE")
    message = spice.job.verify_torch()
    print_result(message=message, context=context, fg="green")
