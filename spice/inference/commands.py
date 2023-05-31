import click

from ..utils.printer import print_result


@click.group()
@click.pass_context
def cli(context):
    """Inference"""
    pass


@cli.command("verify-torch")
@click.pass_context
def verify_torch_command(context):
    """Get System Info"""
    spice = context.obj.get("SPICE")
    message = spice.inference.verify_torch()
    print_result(message=message, context=context, fg="green")


@cli.command("run")
@click.option(
    "--model",
    required=False,
    default="bert-base-uncased",
    show_default="bert-base-uncased",
    help="Model from HuggingFace",
)
@click.option(
    "--input",
    required=False,
    default="spice.cloud is [MASK]!",
    show_default="spice.cloud is [MASK]!",
    help="Input for model to run inference",
)
@click.pass_context
def run_command(context, model: str, input: str):
    """Run HuggingFace Pipeline with Model from hub and a basic input."""
    spice = context.obj.get("SPICE")
    message = spice.inference.run_pipeline(model=model, input=input)
    print_result(message=message, context=context)
