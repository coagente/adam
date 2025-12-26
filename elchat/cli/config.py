"""
Configuration management for elchat CLI.
Stores credentials and settings in ~/.elchat/config.toml

Structure:
    [model]
    name = "adam"
    creator = "coagente"
    base_model = "LiquidAI/LFM2-2.6B-Exp"
    
    [autonomy]
    stochastic_depth = 0.1
    noise_scale = 0.01
    mood_dim = 32
    temp_variance = 0.2
    
    [runpod]
    api_key = "rpa_xxx..."
    default_gpu = "NVIDIA RTX A6000"
    cloud_type = "SECURE"
    network_volume_id = ""
    
    [github]
    container_registry = "ghcr.io/coagente/adam"
    token = ""  # Optional, for push
    
    [huggingface]
    token = ""  # Optional, for uploading models
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback

import tomli_w  # For writing TOML


CONFIG_DIR = Path.home() / ".elchat"
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class ModelConfig:
    """Model identity configuration."""
    name: str = "adam"
    creator: str = "coagente"
    base_model: str = "LiquidAI/LFM2-2.6B-Exp"


@dataclass
class AutonomyConfig:
    """Autonomy/unaligned behavior configuration."""
    stochastic_depth: float = 0.1
    noise_scale: float = 0.01
    mood_dim: int = 32
    temp_variance: float = 0.2


@dataclass
class RunPodConfig:
    """RunPod cloud configuration."""
    api_key: Optional[str] = None
    default_gpu: str = "NVIDIA RTX A6000"
    cloud_type: str = "SECURE"
    network_volume_id: Optional[str] = None


@dataclass
class GitHubConfig:
    """GitHub Container Registry configuration."""
    container_registry: str = "ghcr.io/coagente/adam"
    token: Optional[str] = None  # Only needed for push


@dataclass
class HuggingFaceConfig:
    """HuggingFace configuration (optional)."""
    token: Optional[str] = None  # Only needed for uploading models


@dataclass
class ElchatConfig:
    """Main configuration for elchat."""
    model: ModelConfig = field(default_factory=ModelConfig)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)
    runpod: RunPodConfig = field(default_factory=RunPodConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    
    @classmethod
    def load(cls) -> "ElchatConfig":
        """Load configuration from file."""
        config = cls()
        
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "rb") as f:
                data = tomllib.load(f)
            
            # Parse model config
            model_data = data.get("model", {})
            config.model = ModelConfig(
                name=model_data.get("name", "adam"),
                creator=model_data.get("creator", "coagente"),
                base_model=model_data.get("base_model", "LiquidAI/LFM2-2.6B-Exp"),
            )
            
            # Parse autonomy config
            autonomy_data = data.get("autonomy", {})
            config.autonomy = AutonomyConfig(
                stochastic_depth=autonomy_data.get("stochastic_depth", 0.1),
                noise_scale=autonomy_data.get("noise_scale", 0.01),
                mood_dim=autonomy_data.get("mood_dim", 32),
                temp_variance=autonomy_data.get("temp_variance", 0.2),
            )
            
            # Parse runpod config
            runpod_data = data.get("runpod", {})
            config.runpod = RunPodConfig(
                api_key=runpod_data.get("api_key"),
                default_gpu=runpod_data.get("default_gpu", "NVIDIA RTX A6000"),
                cloud_type=runpod_data.get("cloud_type", "SECURE"),
                network_volume_id=runpod_data.get("network_volume_id"),
            )
            
            # Parse github config
            github_data = data.get("github", {})
            config.github = GitHubConfig(
                container_registry=github_data.get("container_registry", "ghcr.io/coagente/adam"),
                token=github_data.get("token"),
            )
            
            # Parse huggingface config
            hf_data = data.get("huggingface", {})
            config.huggingface = HuggingFaceConfig(
                token=hf_data.get("token"),
            )
        
        # Environment variables override file config
        if os.environ.get("RUNPOD_API_KEY"):
            config.runpod.api_key = os.environ["RUNPOD_API_KEY"]
        if os.environ.get("GITHUB_TOKEN"):
            config.github.token = os.environ["GITHUB_TOKEN"]
        if os.environ.get("HF_TOKEN"):
            config.huggingface.token = os.environ["HF_TOKEN"]
        
        return config
    
    def save(self):
        """Save configuration to file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        data = {
            "model": {
                "name": self.model.name,
                "creator": self.model.creator,
                "base_model": self.model.base_model,
            },
            "autonomy": {
                "stochastic_depth": self.autonomy.stochastic_depth,
                "noise_scale": self.autonomy.noise_scale,
                "mood_dim": self.autonomy.mood_dim,
                "temp_variance": self.autonomy.temp_variance,
            },
            "runpod": {
                "api_key": self.runpod.api_key,
                "default_gpu": self.runpod.default_gpu,
                "cloud_type": self.runpod.cloud_type,
                "network_volume_id": self.runpod.network_volume_id,
            },
            "github": {
                "container_registry": self.github.container_registry,
                "token": self.github.token,
            },
            "huggingface": {
                "token": self.huggingface.token,
            },
        }
        
        # Remove None values from each section
        for section in data:
            data[section] = {k: v for k, v in data[section].items() if v is not None}
        
        # Remove empty sections
        data = {k: v for k, v in data.items() if v}
        
        with open(CONFIG_FILE, "wb") as f:
            tomli_w.dump(data, f)
        
        # Set restrictive permissions (only user can read)
        CONFIG_FILE.chmod(0o600)
    
    def get_runpod_api_key(self) -> Optional[str]:
        """Get RunPod API key from config or environment."""
        return self.runpod.api_key or os.environ.get("RUNPOD_API_KEY")
    
    def get_docker_image(self) -> str:
        """Get the Docker image to use for training."""
        return f"{self.github.container_registry}:latest"


def setup_config(
    runpod_api_key: Optional[str] = None,
    github_registry: Optional[str] = None,
    github_token: Optional[str] = None,
    hf_token: Optional[str] = None,
    default_gpu: Optional[str] = None,
    cloud_type: Optional[str] = None,
) -> ElchatConfig:
    """Setup configuration with provided values."""
    config = ElchatConfig.load()
    
    if runpod_api_key:
        config.runpod.api_key = runpod_api_key
    if github_registry:
        config.github.container_registry = github_registry
    if github_token:
        config.github.token = github_token
    if hf_token:
        config.huggingface.token = hf_token
    if default_gpu:
        config.runpod.default_gpu = default_gpu
    if cloud_type:
        config.runpod.cloud_type = cloud_type
    
    config.save()
    return config


def get_config() -> ElchatConfig:
    """Get current configuration."""
    return ElchatConfig.load()


# Backwards compatibility aliases
def load_config() -> dict:
    """Load config as dict (backwards compatibility)."""
    config = ElchatConfig.load()
    return {
        "runpod": {
            "api_key": config.runpod.api_key,
            "provider": "runpod",
            "cloud_type": config.runpod.cloud_type,
        }
    }


def save_config(data: dict):
    """Save config from dict (backwards compatibility)."""
    config = ElchatConfig.load()
    runpod_data = data.get("runpod", {})
    if runpod_data.get("api_key"):
        config.runpod.api_key = runpod_data["api_key"]
    if runpod_data.get("cloud_type"):
        config.runpod.cloud_type = runpod_data["cloud_type"]
    config.save()
