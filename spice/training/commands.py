import click

from ..utils.printer import print_result


@click.group()
@click.pass_context
def cli(context):
    """Inference"""
    pass


@cli.command("worker")
@click.option(
    "--verify",
    is_flag=True,
    required=False,
    help="Use the worker to verify training rounds",
)
@click.pass_context
def worker_command(context, verify: bool):
    """Run as a worker that picks up tasks from the spice.cloud queue."""
    spice = context.obj.get("SPICE")
    message = spice.training.worker(verify)
    print_result(message=message, context=context)
