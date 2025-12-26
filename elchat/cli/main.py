"""
elchat CLI - Entrenar LLMs en la nube por ~$10

Usage:
    elchat config [--runpod-key KEY]
    elchat train [--config CONFIG] [--dry-run]
    elchat status
    elchat logs [-f]
    elchat stop
"""

import typer
from rich.console import Console

from elchat.cli.commands import train, status, logs, stop, config_cmd

app = typer.Typer(
    name="elchat",
    help="Unaligned LLM - Entrena tu propio modelo autónomo",
    add_completion=False,
)
console = Console()

# Register commands
app.add_typer(config_cmd.app, name="config")
app.add_typer(train.app, name="train")
app.command()(status.status)
app.command()(logs.logs)
app.command()(stop.stop)


@app.callback()
def callback():
    """
    elchat CLI - Entrenar tu propio LLM en español por ~$10
    """
    pass


def main():
    app()


if __name__ == "__main__":
    main()

