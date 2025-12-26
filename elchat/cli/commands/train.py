"""
Train command - Create pod and start training on RunPod.

Uses pre-built Docker image from ghcr.io/coagente/adam instead of SSH/rsync.
The container auto-starts training based on environment variables.
"""

import os
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from elchat.cli.runpod_client import RunPodClient
from elchat.cli.config import get_config, ElchatConfig

app = typer.Typer(help="Entrenar el modelo Adam en RunPod")
console = Console()

# Default config path
DEFAULT_CONFIG = "configs/adam.yaml"

# Default Docker image
DEFAULT_IMAGE = "ghcr.io/coagente/adam:latest"


class SmokeTestResult:
    """Result of a smoke test."""
    def __init__(self, name: str, passed: bool, message: str = "", details: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details


def run_smoke_tests(config: dict) -> list[SmokeTestResult]:
    """Run pre-flight smoke tests before training.
    
    Tests:
    1. API key configured
    2. API connection works
    3. GPU available
    4. Cost estimate
    5. Config file valid
    """
    results = []
    
    # Test 1: API Key
    app_config = get_config()
    api_key = app_config.get_runpod_api_key()
    if api_key:
        results.append(SmokeTestResult(
            "API Key", True, 
            "Configurada",
            f"{api_key[:8]}...{api_key[-4:]}"
        ))
    else:
        results.append(SmokeTestResult(
            "API Key", False,
            "No configurada",
            "Usa: elchat config set --runpod-key TU_KEY"
        ))
        return results  # Can't continue without API key
    
    # Test 2: Connection
    try:
        client = RunPodClient()
        if client.validate_connection():
            results.append(SmokeTestResult("Conexión RunPod", True, "OK"))
        else:
            results.append(SmokeTestResult("Conexión RunPod", False, "Falló"))
            return results
    except Exception as e:
        results.append(SmokeTestResult("Conexión RunPod", False, str(e)))
        return results
    
    # Test 3: GPU Availability
    gpu_type = config["cloud"]["gpu"]
    cloud_type = config["cloud"].get("cloud_type", "SECURE")
    available, price = client.check_gpu_availability(gpu_type, cloud_type)
    
    if available and price:
        results.append(SmokeTestResult(
            "GPU Disponible", True,
            f"{gpu_type}",
            f"${price}/hr ({cloud_type})"
        ))
    else:
        results.append(SmokeTestResult(
            "GPU Disponible", False,
            f"{gpu_type} no disponible en {cloud_type}",
            "Intenta con otra GPU o cloud type"
        ))
    
    # Test 4: Cost Estimate
    if available and price:
        gpu_count = config["cloud"].get("gpu_count", 1)
        hours = config["estimate"]["hours"]
        estimated_cost = client.estimate_cost(gpu_type, gpu_count, hours, cloud_type)
        config_cost = config["estimate"]["cost_usd"]
        
        if estimated_cost:
            diff = abs(estimated_cost - config_cost)
            if diff < config_cost * 0.5:  # Within 50%
                results.append(SmokeTestResult(
                    "Costo Estimado", True,
                    f"${estimated_cost:.2f}",
                    f"Config dice ${config_cost:.2f}"
                ))
            else:
                results.append(SmokeTestResult(
                    "Costo Estimado", True,
                    f"${estimated_cost:.2f}",
                    f"[yellow]Config dice ${config_cost:.2f} (diferencia)[/yellow]"
                ))
    
    # Test 5: Config Valid
    required_fields = ["cloud.gpu", "training.base_model", "data.num_shards"]
    missing = []
    for field in required_fields:
        parts = field.split(".")
        val = config
        for p in parts:
            val = val.get(p) if isinstance(val, dict) else None
        if val is None:
            missing.append(field)
    
    if not missing:
        results.append(SmokeTestResult("Config Válida", True, "Completa"))
    else:
        results.append(SmokeTestResult(
            "Config Válida", False,
            "Campos faltantes",
            ", ".join(missing)
        ))
    
    return results


def show_smoke_test_results(results: list[SmokeTestResult]) -> bool:
    """Display smoke test results. Returns True if all passed."""
    console.print()
    console.print("[bold]Smoke Tests:[/bold]")
    
    all_passed = True
    for r in results:
        icon = "[green]✓[/green]" if r.passed else "[red]✗[/red]"
        msg = f"  {icon} {r.name}: {r.message}"
        if r.details:
            msg += f" [dim]({r.details})[/dim]"
        console.print(msg)
        if not r.passed:
            all_passed = False
    
    console.print()
    return all_passed


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
    
    gpu_count = config["cloud"].get("gpu_count", 1)
    cloud_type = config["cloud"].get("cloud_type", "SECURE")
    training = config["training"]
    
    # Model identity from global config
    table.add_row("Modelo", f"[bold]{app_config.model.name}[/bold] by {app_config.model.creator}")
    table.add_row("Modelo base", training.get("base_model", app_config.model.base_model))
    table.add_row("GPU", config["cloud"]["gpu"])
    table.add_row("Cantidad GPUs", str(gpu_count))
    table.add_row("Cloud", cloud_type)
    table.add_row("Volumen", f"{config['cloud']['volume_gb']} GB")
    table.add_row("Tokens objetivo", f"{training['target_tokens']:,}")
    table.add_row("Batch size", str(training.get("device_batch_size", 4)))
    table.add_row("Shards de datos", str(config["data"]["num_shards"]))
    table.add_row("Costo estimado", f"${config['estimate']['cost_usd']:.2f}")
    table.add_row("Tiempo estimado", f"{config['estimate']['hours']} horas")
    
    # Docker image
    docker_image = config.get("docker", {}).get("image", app_config.get_docker_image())
    table.add_row("Docker Image", docker_image)
    
    # Mostrar parámetros de autonomía si están activos
    autonomy_active = any([
        training.get("stochastic_depth", 0) > 0,
        training.get("noise_scale", 0) > 0,
        training.get("mood_dim", 0) > 0,
        training.get("stochastic_temp_var", 0) > 0,
    ])
    
    if autonomy_active:
        table.add_row("", "")  # Separador
        table.add_row("[bold yellow]Autonomía[/bold yellow]", "[bold yellow]Activa[/bold yellow]")
        if training.get("stochastic_depth", 0) > 0:
            table.add_row("  Stochastic Depth", f"{training['stochastic_depth']}")
        if training.get("noise_scale", 0) > 0:
            table.add_row("  Noise Scale", f"{training['noise_scale']}")
        if training.get("mood_dim", 0) > 0:
            table.add_row("  Mood Dim", f"{training['mood_dim']}")
        if training.get("stochastic_temp_var", 0) > 0:
            table.add_row("  Temp Variance", f"{training['stochastic_temp_var']}")
    
    panel = Panel(table, title="[bold blue]Adam - Entrenamiento en RunPod[/bold blue]")
    console.print(panel)


def build_env_vars(config: dict, app_config: ElchatConfig) -> dict:
    """Build environment variables for the Docker container."""
    training = config.get("training", {})
    
    env_vars = {
        # Model
        "BASE_MODEL": training.get("base_model", app_config.model.base_model),
        "TARGET_TOKENS": str(training.get("target_tokens", 50_000_000)),
        "DEVICE_BATCH_SIZE": str(training.get("device_batch_size", 4)),
        "NUM_SHARDS": str(config.get("data", {}).get("num_shards", 3)),
        
        # Identity
        "MODEL_NAME": app_config.model.name,
        "MODEL_CREATOR": app_config.model.creator,
        
        # Autonomy parameters
        "STOCHASTIC_DEPTH": str(training.get("stochastic_depth", 0)),
        "NOISE_SCALE": str(training.get("noise_scale", 0)),
        "MOOD_DIM": str(training.get("mood_dim", 0)),
        "STOCHASTIC_TEMP_VAR": str(training.get("stochastic_temp_var", 0)),
    }
    
    # Add HuggingFace token if configured
    if app_config.huggingface.token:
        env_vars["HF_TOKEN"] = app_config.huggingface.token
    
    return env_vars


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
    skip_tests: bool = typer.Option(
        False,
        "--skip-tests",
        help="Saltar smoke tests (no recomendado)"
    ),
    yes: bool = typer.Option(
        False,
        "--yes", "-y",
        help="No pedir confirmación"
    ),
    image: Optional[str] = typer.Option(
        None,
        "--image", "-i",
        help="Override Docker image (default: from config)"
    ),
):
    """
    Entrenar el modelo Adam en RunPod.
    
    Crea un pod GPU con la imagen Docker pre-construida y ejecuta el entrenamiento.
    Sin necesidad de sincronizar código - todo está en la imagen.
    
    Ejemplo:
        elchat train --config configs/adam.yaml --dry-run
    """
    # Load configs
    app_config = get_config()
    training_config = load_config(config)
    
    # Show training info
    show_training_info(training_config, app_config)
    
    # Run smoke tests unless skipped
    if not skip_tests:
        results = run_smoke_tests(training_config)
        all_passed = show_smoke_test_results(results)
        
        if not all_passed:
            console.print("[red]Smoke tests fallaron. Corrige los errores antes de continuar.[/red]")
            console.print("[dim]Usa --skip-tests para saltar (no recomendado)[/dim]")
            raise typer.Exit(1)
    
    if dry_run:
        console.print("[bold green]✓ Dry-run completado[/bold green]")
        console.print("[dim]Todo listo para entrenar. Quita --dry-run para ejecutar.[/dim]")
        return
    
    # Confirm with user
    if not yes:
        if not typer.confirm("¿Iniciar entrenamiento?"):
            console.print("[yellow]Cancelado[/yellow]")
            raise typer.Exit(0)
    
    console.print()
    
    try:
        client = RunPodClient()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Configura con: elchat config set --runpod-key TU_KEY[/dim]")
        raise typer.Exit(1)
    
    # Determine Docker image
    docker_image = image or training_config.get("docker", {}).get("image") or app_config.get_docker_image()
    
    # Build environment variables
    env_vars = build_env_vars(training_config, app_config)
    
    pod_name = training_config.get("name", f"adam-{app_config.model.name}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Create pod with Docker image
        task = progress.add_task("[1/3] Creando pod en RunPod...", total=None)
        
        # Check if pod already exists
        existing = client.get_pod_by_name(pod_name)
        if existing:
            console.print(f"[yellow]Pod '{pod_name}' ya existe (ID: {existing.id})[/yellow]")
            if existing.status == "RUNNING":
                console.print("[yellow]El pod ya está corriendo - probablemente entrenando[/yellow]")
                console.print(f"[dim]SSH: ssh -p {existing.ssh_port} root@{existing.ssh_host}[/dim]")
                console.print("[dim]Para terminar: elchat stop[/dim]")
                raise typer.Exit(0)
            pod = existing
        else:
            gpu_count = training_config["cloud"].get("gpu_count", 1)
            cloud_type = training_config["cloud"].get("cloud_type", "SECURE")
            
            pod = client.create_pod(
                name=pod_name,
                gpu_type=training_config["cloud"]["gpu"],
                volume_gb=training_config["cloud"]["volume_gb"],
                cloud_type=cloud_type,
                gpu_count=gpu_count,
                image=docker_image,
                env=env_vars,
            )
            console.print(f"[green]✓[/green] Pod creado: {pod.id}")
            console.print(f"[dim]  GPU: {gpu_count}x {training_config['cloud']['gpu']}[/dim]")
            console.print(f"[dim]  Cloud: {cloud_type}[/dim]")
            console.print(f"[dim]  Image: {docker_image}[/dim]")
        
        progress.update(task, completed=True)
        
        # Step 2: Wait for pod to be ready
        task = progress.add_task("[2/3] Esperando que el pod inicie...", total=None)
        pod = client.wait_for_pod(pod.id)
        console.print(f"[green]✓[/green] Pod corriendo: {pod.ssh_host}:{pod.ssh_port}")
        progress.update(task, completed=True)
        
        # Step 3: Training starts automatically via Docker entrypoint
        task = progress.add_task("[3/3] Entrenamiento iniciado (en container)...", total=None)
        console.print()
        console.print("[bold]Entrenamiento en progreso...[/bold]")
        console.print("[dim]El container ejecuta el entrenamiento automáticamente.[/dim]")
        console.print("[dim]Usa 'elchat logs' para ver el progreso.[/dim]")
        progress.update(task, completed=True)
    
    console.print()
    console.print(Panel(
        f"[bold green]¡Entrenamiento iniciado![/bold green]\n\n"
        f"Pod ID: {pod.id}\n"
        f"SSH: ssh -p {pod.ssh_port} root@{pod.ssh_host}\n\n"
        f"Modelo: [bold]{app_config.model.name}[/bold] by {app_config.model.creator}\n"
        f"Base: {training_config['training'].get('base_model', app_config.model.base_model)}\n\n"
        "Comandos útiles:\n"
        "  [cyan]elchat logs[/cyan]     - Ver logs del entrenamiento\n"
        "  [cyan]elchat status[/cyan]   - Estado del pod\n"
        "  [cyan]elchat stop[/cyan]     - Detener el pod",
        title="Adam - Entrenamiento"
    ))
