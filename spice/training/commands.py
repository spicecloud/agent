import click

from ..utils.printer import print_result


@click.group()
@click.pass_context
def cli(context):
    """Training"""
    pass
