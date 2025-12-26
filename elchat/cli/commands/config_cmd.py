"""
Config command - Manage elchat configuration.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from elchat.cli.config import get_config, setup_config, CONFIG_FILE, ElchatConfig


app = typer.Typer(help="Gestionar configuración de elchat")
console = Console()


@app.command("set")
def config_set(
    runpod_key: Optional[str] = typer.Option(None, "--runpod-key", "-r", help="RunPod API key"),
    github_registry: Optional[str] = typer.Option(None, "--github-registry", "-g", help="GitHub Container Registry (e.g. ghcr.io/user/repo)"),
    github_token: Optional[str] = typer.Option(None, "--github-token", help="GitHub token (for pushing images)"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", help="HuggingFace token (optional)"),
    default_gpu: Optional[str] = typer.Option(None, "--default-gpu", help="Default GPU type"),
    cloud_type: Optional[str] = typer.Option(None, "--cloud-type", "-c", help="Default cloud type (COMMUNITY/SECURE)"),
):
    """Configurar credenciales y opciones."""
    config = setup_config(
        runpod_api_key=runpod_key,
        github_registry=github_registry,
        github_token=github_token,
        hf_token=hf_token,
        default_gpu=default_gpu,
        cloud_type=cloud_type.upper() if cloud_type else None,
    )
    
    changes = []
    if runpod_key:
        changes.append("RunPod API key")
    if github_registry:
        changes.append(f"GitHub registry: {github_registry}")
    if github_token:
        changes.append("GitHub token")
    if hf_token:
        changes.append("HuggingFace token")
    if default_gpu:
        changes.append(f"Default GPU: {default_gpu}")
    if cloud_type:
        changes.append(f"Cloud type: {cloud_type.upper()}")
    
    if changes:
        for change in changes:
            console.print(f"[green]✓[/green] {change}")
        console.print(f"\n[dim]Configuración guardada en {CONFIG_FILE}[/dim]")
    else:
        console.print("[yellow]No se especificaron opciones.[/yellow]")
        console.print("Usa --help para ver opciones disponibles.")


@app.command("show")
def config_show():
    """Mostrar configuración actual."""
    config = get_config()
    
    table = Table(title="Configuración Adam", show_header=False)
    table.add_column("Campo", style="cyan")
    table.add_column("Valor", style="green")
    
    # Model section
    table.add_row("[bold]Modelo[/bold]", "")
    table.add_row("  Nombre", config.model.name)
    table.add_row("  Creador", config.model.creator)
    table.add_row("  Base", config.model.base_model)
    
    # Autonomy section
    table.add_row("", "")
    table.add_row("[bold]Autonomía[/bold]", "")
    table.add_row("  Stochastic Depth", str(config.autonomy.stochastic_depth))
    table.add_row("  Noise Scale", str(config.autonomy.noise_scale))
    table.add_row("  Mood Dim", str(config.autonomy.mood_dim))
    table.add_row("  Temp Variance", str(config.autonomy.temp_variance))
    
    # RunPod section
    table.add_row("", "")
    table.add_row("[bold]RunPod[/bold]", "")
    api_key = config.runpod.api_key
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        table.add_row("  API Key", masked)
    else:
        table.add_row("  API Key", "[red]No configurada[/red]")
    table.add_row("  Default GPU", config.runpod.default_gpu)
    table.add_row("  Cloud Type", config.runpod.cloud_type)
    
    # GitHub section
    table.add_row("", "")
    table.add_row("[bold]GitHub[/bold]", "")
    table.add_row("  Registry", config.github.container_registry)
    table.add_row("  Token", "[green]Configurado[/green]" if config.github.token else "[dim]No configurado[/dim]")
    
    # HuggingFace section
    table.add_row("", "")
    table.add_row("[bold]HuggingFace[/bold]", "")
    table.add_row("  Token", "[green]Configurado[/green]" if config.huggingface.token else "[dim]No configurado (opcional)[/dim]")
    
    # File location
    table.add_row("", "")
    table.add_row("[dim]Config File[/dim]", str(CONFIG_FILE))
    
    panel = Panel(table, title="[bold blue]Adam - Configuración[/bold blue]")
    console.print(panel)


@app.command("test")
def config_test():
    """Probar conexión con RunPod y mostrar GPUs disponibles."""
    from elchat.cli.runpod_client import RunPodClient
    
    config = get_config()
    
    console.print("[bold]Probando configuración...[/bold]\n")
    
    # Test RunPod
    console.print("RunPod: ", end="")
    try:
        client = RunPodClient()
        if client.validate_connection():
            console.print("[green]✓ Conexión exitosa[/green]")
        else:
            console.print("[red]✗ Falló[/red]")
            raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)
    
    # Show available GPUs
    console.print("\n[bold]GPUs disponibles (SECURE):[/bold]")
    gpu_types = [
        "NVIDIA RTX A6000",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100 40GB PCIe",
    ]
    
    for gpu_type in gpu_types:
        info = client.get_gpu_info(gpu_type)
        if info:
            price = info.secure_price or info.community_price
            if price:
                cloud = "SECURE" if info.secure_price else "COMMUNITY"
                console.print(f"  • {info.name}: ${price:.2f}/hr ({info.vram_gb}GB) [{cloud}]")
    
    # Test Docker image accessibility
    console.print(f"\n[bold]Docker Image:[/bold]")
    console.print(f"  {config.get_docker_image()}")
    console.print("  [dim](La imagen se descarga automáticamente al crear el pod)[/dim]")


@app.command("init")
def config_init(
    runpod_key: str = typer.Option(..., "--runpod-key", "-r", prompt="RunPod API Key", help="RunPod API key"),
):
    """Configuración inicial interactiva."""
    console.print("\n[bold]Configurando Adam...[/bold]\n")
    
    # Save RunPod key
    config = setup_config(runpod_api_key=runpod_key)
    console.print("[green]✓[/green] RunPod API key configurada")
    
    # Test connection
    from elchat.cli.runpod_client import RunPodClient
    try:
        client = RunPodClient()
        if client.validate_connection():
            console.print("[green]✓[/green] Conexión con RunPod verificada")
        else:
            console.print("[yellow]⚠[/yellow] No se pudo verificar la conexión")
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Error de conexión: {e}")
    
    console.print(f"\n[dim]Configuración guardada en {CONFIG_FILE}[/dim]")
    console.print("\n[bold green]¡Listo![/bold green] Puedes entrenar con:")
    console.print("  [cyan]elchat train --config configs/adam.yaml --dry-run[/cyan]")


@app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context):
    """Gestionar configuración de elchat."""
    if ctx.invoked_subcommand is None:
        config_show()
