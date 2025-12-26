"""
Status command - Show status of training pods.
"""

import typer
from rich.console import Console
from rich.table import Table

from elchat.cli.runpod_client import RunPodClient

console = Console()


def status():
    """
    Mostrar el estado de los pods de entrenamiento.
    """
    try:
        client = RunPodClient()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Configura RUNPOD_API_KEY en tu entorno[/dim]")
        raise typer.Exit(1)
    
    pods = client.list_pods()
    
    if not pods:
        console.print("[yellow]No hay pods activos[/yellow]")
        return
    
    table = Table(title="Pods de RunPod")
    table.add_column("ID", style="cyan")
    table.add_column("Nombre", style="green")
    table.add_column("Estado", style="yellow")
    table.add_column("GPU", style="magenta")
    table.add_column("SSH", style="dim")
    
    for pod in pods:
        ssh_info = f"{pod.ssh_host}:{pod.ssh_port}" if pod.ssh_host else "-"
        
        # Color status
        status_style = {
            "RUNNING": "[green]RUNNING[/green]",
            "PENDING": "[yellow]PENDING[/yellow]",
            "STOPPED": "[red]STOPPED[/red]",
        }.get(pod.status, pod.status)
        
        table.add_row(
            pod.id,
            pod.name,
            status_style,
            pod.gpu_type,
            ssh_info,
        )
    
    console.print(table)

