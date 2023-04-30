import click

from .auth.commands import cli as auth_cli


@click.group()
@click.option(
    "--debug", is_flag=True, show_default=True, default=False, help="Enable debug mode."
)
@click.option(
    "--json",
    is_flag=True,
    show_default=True,
    default=False,
    help="Turn all outputs into JSON.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity of output levels.",
)
@click.pass_context
def cli(context, debug, json, verbose):
    context.ensure_object(dict)
    context.obj["DEBUG"] = debug
    context.obj["JSON"] = json
    context.obj["VERBOSE"] = verbose


cli.add_command(auth_cli, "auth")


if __name__ == "__main__":
    cli(obj={})
