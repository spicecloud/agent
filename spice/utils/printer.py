import json

import click


def print_result(message: str, context: click.Context, fg: str = None):
    """Print message to stdout or stderr"""
    if context.obj["JSON"]:
        message = json.dumps(message, indent=2)
    click.secho(message, fg=fg)
