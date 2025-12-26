"""
Stop command - Stop or terminate training pods.
"""

import typer
from rich.console import Console

from elchat.cli.runpod_client import RunPodClient

console = Console()


def stop(
    pod_name: str = typer.Option(
        "elchat-training",
        "--pod", "-p",
        help="Nombre del pod"
    ),
    terminate: bool = typer.Option(
        False,
        "--terminate", "-t",
        help="Terminar el pod completamente (eliminar)"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="No pedir confirmación"
    ),
):
    """
    Detener o terminar un pod de entrenamiento.
    """
    try:
        client = RunPodClient()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Configura RUNPOD_API_KEY en tu entorno[/dim]")
        raise typer.Exit(1)
    
    pod = client.get_pod_by_name(pod_name)
    
    if not pod:
        console.print(f"[red]Pod '{pod_name}' no encontrado[/red]")
        raise typer.Exit(1)
    
    action = "terminar" if terminate else "detener"
    
    if not force:
        console.print(f"Pod: [cyan]{pod.name}[/cyan] (ID: {pod.id})")
        console.print(f"Estado: [yellow]{pod.status}[/yellow]")
        console.print(f"GPU: {pod.gpu_type}")
        console.print()
        
        if not typer.confirm(f"¿{action.capitalize()} este pod?"):
            console.print("[yellow]Cancelado[/yellow]")
            raise typer.Exit(0)
    
    if terminate:
        success = client.terminate_pod(pod.id)
        if success:
            console.print(f"[green]✓ Pod terminado: {pod.id}[/green]")
        else:
            console.print(f"[red]Error al terminar pod[/red]")
            raise typer.Exit(1)
    else:
        success = client.stop_pod(pod.id)
        if success:
            console.print(f"[green]✓ Pod detenido: {pod.id}[/green]")
            console.print("[dim]El pod puede reiniciarse más tarde sin perder datos[/dim]")
        else:
            console.print(f"[red]Error al detener pod[/red]")
            raise typer.Exit(1)

