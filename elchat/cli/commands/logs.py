"""
Logs command - View training logs from pod.
"""

import subprocess
import sys

import typer
from rich.console import Console

from elchat.cli.runpod_client import RunPodClient

console = Console()


def logs(
    follow: bool = typer.Option(
        False,
        "--follow", "-f",
        help="Seguir los logs en tiempo real"
    ),
    pod_name: str = typer.Option(
        "elchat-training",
        "--pod", "-p",
        help="Nombre del pod"
    ),
    lines: int = typer.Option(
        100,
        "--lines", "-n",
        help="Número de líneas a mostrar"
    ),
):
    """
    Ver logs del entrenamiento.
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
    
    if not pod.ssh_host:
        console.print(f"[yellow]Pod '{pod_name}' no está listo aún[/yellow]")
        raise typer.Exit(1)
    
    # Build SSH command to tail logs
    tail_cmd = f"tail -n {lines}"
    if follow:
        tail_cmd += " -f"
    
    log_files = "/workspace/elchat/*.log /workspace/elchat/logs/*.log 2>/dev/null || echo 'No hay logs aún'"
    
    ssh_cmd = [
        "ssh", "-p", str(pod.ssh_port),
        "-o", "StrictHostKeyChecking=no",
        f"root@{pod.ssh_host}",
        f"{tail_cmd} {log_files}"
    ]
    
    console.print(f"[dim]Conectando a {pod.ssh_host}:{pod.ssh_port}...[/dim]")
    console.print()
    
    try:
        subprocess.run(ssh_cmd)
    except KeyboardInterrupt:
        console.print("\n[dim]Desconectado[/dim]")

