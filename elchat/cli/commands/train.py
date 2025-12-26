"""
Train command - Start training on RunPod Serverless.

No SSH required - submits job via HTTP API and polls for status.
Checkpoints are saved to HuggingFace Hub automatically.
"""

import os
import time
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.live import Live

from elchat.cli.config import get_config, ElchatConfig
from elchat.cli.serverless_client import ServerlessClient, JobStatus

app = typer.Typer(help="Entrenar el modelo Adam en RunPod Serverless")
console = Console()

DEFAULT_CONFIG = "configs/adam.yaml"


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)
    
    with open(path) as f:
        return yaml.safe_load(f)


def show_training_info(config: dict, app_config: ElchatConfig):
    """Display training configuration summary."""
    table = Table(title="Configuración de Entrenamiento", show_header=False)
    table.add_column("Campo", style="cyan")
    table.add_column("Valor", style="green")
    
    training = config.get("training", {})
    checkpoints = config.get("checkpoints", {})
    serverless = config.get("serverless", {})
    
    table.add_row("Modelo", f"[bold]{app_config.model.name}[/bold] by {app_config.model.creator}")
    table.add_row("Modelo base", training.get("base_model", ""))
    table.add_row("Modo", "[green]RunPod Serverless[/green]")
    table.add_row("GPU", config.get("cloud", {}).get("gpu", "A100 80GB"))
    table.add_row("Tokens objetivo", f"{training.get('target_tokens', 0):,}")
    table.add_row("Batch size", f"{training.get('device_batch_size', 1)} x {training.get('gradient_accumulation_steps', 32)} accum")
    table.add_row("Gradient Checkpointing", "[green]Habilitado[/green]" if training.get("gradient_checkpointing") else "Deshabilitado")
    table.add_row("Checkpoints", f"Cada {checkpoints.get('every', 100)} steps -> HuggingFace")
    table.add_row("Costo estimado", f"${config.get('estimate', {}).get('cost_usd', 0):.2f}")
    table.add_row("Tiempo estimado", f"{config.get('estimate', {}).get('hours', 0)} horas")
    
    # Endpoint
    endpoint_id = serverless.get("endpoint_id")
    if endpoint_id:
        table.add_row("Endpoint ID", endpoint_id)
    else:
        table.add_row("Endpoint ID", "[yellow]No configurado[/yellow]")
    
    # Autonomy params
    autonomy_active = any([
        training.get("stochastic_depth", 0) > 0,
        training.get("noise_scale", 0) > 0,
        training.get("mood_dim", 0) > 0,
        training.get("stochastic_temp_var", 0) > 0,
    ])
    
    if autonomy_active:
        table.add_row("", "")
        table.add_row("[bold yellow]Autonomía[/bold yellow]", "[bold yellow]Activa[/bold yellow]")
    
    panel = Panel(table, title="[bold blue]Adam - RunPod Serverless Training[/bold blue]")
    console.print(panel)


def run_smoke_tests(config: dict, app_config: ElchatConfig) -> bool:
    """Run pre-flight checks."""
    console.print("\n[bold]Smoke Tests:[/bold]")
    all_passed = True
    
    # Test 1: API Key
    api_key = app_config.get_runpod_api_key()
    if api_key:
        console.print(f"  [green]✓[/green] API Key: Configurada ({api_key[:8]}...)")
    else:
        console.print("  [red]✗[/red] API Key: No configurada")
        all_passed = False
    
    # Test 2: Endpoint ID
    endpoint_id = config.get("serverless", {}).get("endpoint_id")
    if endpoint_id:
        console.print(f"  [green]✓[/green] Endpoint ID: {endpoint_id}")
    else:
        console.print("  [yellow]![/yellow] Endpoint ID: No configurado (se creará manualmente)")
    
    # Test 3: HuggingFace token (for checkpoints)
    hf_token = app_config.huggingface.token
    if hf_token:
        console.print(f"  [green]✓[/green] HuggingFace Token: Configurado")
    else:
        console.print("  [yellow]![/yellow] HuggingFace Token: No configurado (checkpoints locales)")
    
    # Test 4: Config valid
    required = ["training.base_model", "data.num_shards"]
    missing = []
    for field in required:
        parts = field.split(".")
        val = config
        for p in parts:
            val = val.get(p) if isinstance(val, dict) else None
        if val is None:
            missing.append(field)
    
    if not missing:
        console.print("  [green]✓[/green] Config: Válida")
    else:
        console.print(f"  [red]✗[/red] Config: Faltan campos: {', '.join(missing)}")
        all_passed = False
    
    console.print()
    return all_passed


def build_job_input(config: dict, app_config: ElchatConfig) -> dict:
    """Build the job input for the serverless endpoint."""
    training = config.get("training", {})
    checkpoints = config.get("checkpoints", {})
    data = config.get("data", {})
    
    job_input = {
        "base_model": training.get("base_model", app_config.model.base_model),
        "target_tokens": training.get("target_tokens", 100_000_000),
        "device_batch_size": training.get("device_batch_size", 1),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 32),
        "checkpoint_every": checkpoints.get("every", 100) if checkpoints.get("enabled", True) else 0,
        "num_shards": data.get("num_shards", 5),
    }
    
    # HuggingFace for checkpoints
    if checkpoints.get("destination") == "huggingface":
        job_input["hf_repo_id"] = checkpoints.get("hf_repo", f"coagente/{app_config.model.name}")
        if app_config.huggingface.token:
            job_input["hf_token"] = app_config.huggingface.token
    
    # Resume from checkpoint
    resume = checkpoints.get("resume_from")
    if resume and resume != "auto":
        job_input["resume_from_checkpoint"] = resume
    
    return job_input


def show_progress(status: JobStatus):
    """Display job progress."""
    status_colors = {
        "IN_QUEUE": "yellow",
        "IN_PROGRESS": "blue",
        "COMPLETED": "green",
        "FAILED": "red",
        "CANCELLED": "red",
    }
    color = status_colors.get(status.status, "white")
    console.print(f"  Status: [{color}]{status.status}[/{color}]", end="")
    
    if status.execution_time:
        console.print(f" | Time: {status.execution_time:.1f}s", end="")
    
    console.print()


@app.callback(invoke_without_command=True)
def train(
    config: str = typer.Option(
        DEFAULT_CONFIG,
        "--config", "-c",
        help="Path to training config YAML file"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run", "-d",
        help="Solo mostrar info y smoke tests, sin ejecutar"
    ),
    endpoint_id: Optional[str] = typer.Option(
        None,
        "--endpoint", "-e",
        help="Override endpoint ID from config"
    ),
    yes: bool = typer.Option(
        False,
        "--yes", "-y",
        help="No pedir confirmación"
    ),
):
    """
    Entrenar el modelo Adam en RunPod Serverless.
    
    Usa imagen Docker pre-construida. Los checkpoints se guardan
    automáticamente en HuggingFace Hub.
    
    Ejemplo:
        elchat train --config configs/adam.yaml --dry-run
    """
    app_config = get_config()
    training_config = load_config(config)
    
    # Override endpoint if provided
    if endpoint_id:
        if "serverless" not in training_config:
            training_config["serverless"] = {}
        training_config["serverless"]["endpoint_id"] = endpoint_id
    
    show_training_info(training_config, app_config)
    
    all_passed = run_smoke_tests(training_config, app_config)
    
    if not all_passed:
        console.print("[red]Smoke tests fallaron. Corrige los errores antes de continuar.[/red]")
        raise typer.Exit(1)
    
    # Check endpoint
    final_endpoint = training_config.get("serverless", {}).get("endpoint_id")
    if not final_endpoint:
        console.print("[yellow]No hay endpoint configurado.[/yellow]")
        console.print()
        console.print("Para crear un endpoint:")
        console.print("  1. Ve a https://runpod.io/console/serverless")
        console.print("  2. Crea un nuevo endpoint con imagen: ghcr.io/coagente/adam-worker:latest")
        console.print("  3. Copia el Endpoint ID")
        console.print("  4. Ejecuta: elchat train --endpoint YOUR_ENDPOINT_ID")
        console.print()
        console.print("O agrega a configs/adam.yaml:")
        console.print("  serverless:")
        console.print("    endpoint_id: YOUR_ENDPOINT_ID")
        raise typer.Exit(1)
    
    if dry_run:
        console.print("[bold green]✓ Dry-run completado[/bold green]")
        console.print("[dim]Todo listo. Quita --dry-run para ejecutar.[/dim]")
        
        # Show job input preview
        job_input = build_job_input(training_config, app_config)
        console.print("\n[dim]Job input:[/dim]")
        for k, v in job_input.items():
            if k != "hf_token":  # Don't show token
                console.print(f"  [dim]{k}: {v}[/dim]")
        return
    
    if not yes:
        if not typer.confirm("¿Iniciar entrenamiento?"):
            console.print("[yellow]Cancelado[/yellow]")
            raise typer.Exit(0)
    
    console.print()
    
    # Initialize client
    try:
        client = ServerlessClient(endpoint_id=final_endpoint)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    # Build job input
    job_input = build_job_input(training_config, app_config)
    
    # Submit job
    console.print("[bold]Enviando job a RunPod Serverless...[/bold]")
    
    try:
        job_id = client.submit_job(job_input)
        console.print(f"[green]✓[/green] Job enviado: {job_id}")
    except Exception as e:
        console.print(f"[red]Error enviando job: {e}[/red]")
        raise typer.Exit(1)
    
    # Wait for completion with progress display
    console.print()
    console.print("[bold]Esperando resultado...[/bold]")
    console.print("[dim]Ctrl+C para cancelar (el job seguirá corriendo)[/dim]")
    console.print()
    
    try:
        last_status = None
        while True:
            status = client.get_job_status(job_id)
            
            if status.status != last_status:
                show_progress(status)
                last_status = status.status
            
            if status.status in ["COMPLETED", "FAILED", "CANCELLED"]:
                break
            
            time.sleep(10)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrumpido. El job sigue corriendo en RunPod.[/yellow]")
        console.print(f"Para verificar status: elchat status --job {job_id}")
        raise typer.Exit(0)
    
    # Show result
    console.print()
    
    if status.status == "COMPLETED":
        output = status.output or {}
        console.print(Panel(
            f"[bold green]¡Entrenamiento completado![/bold green]\n\n"
            f"Steps: {output.get('steps', 'N/A')}\n"
            f"Loss final: {output.get('final_loss', 'N/A')}\n"
            f"Tiempo: {output.get('training_time_minutes', 0):.1f} minutos\n"
            f"Modelo: {output.get('model_repo', 'N/A')}\n\n"
            f"Job ID: {job_id}",
            title="Adam - Training Complete"
        ))
    else:
        console.print(Panel(
            f"[bold red]Entrenamiento falló[/bold red]\n\n"
            f"Status: {status.status}\n"
            f"Error: {status.error or 'Unknown'}\n\n"
            f"Job ID: {job_id}",
            title="Adam - Training Failed"
        ))
        raise typer.Exit(1)
