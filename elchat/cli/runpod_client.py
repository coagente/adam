"""
RunPod API client for elchat.
Handles pod creation, management, and file transfers.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import runpod

from elchat.cli.config import get_config


@dataclass
class PodInfo:
    """Information about a RunPod pod."""
    id: str
    name: str
    status: str
    gpu_type: str
    gpu_count: int = 1
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None


@dataclass 
class GPUInfo:
    """Information about a GPU type."""
    id: str
    name: str
    vram_gb: int
    community_price: Optional[float] = None
    secure_price: Optional[float] = None
    spot_price: Optional[float] = None
    available: bool = True


class RunPodClient:
    """Client for interacting with RunPod API."""

    def __init__(self, api_key: Optional[str] = None):
        # Try to get API key from: argument > config file > environment
        if api_key:
            self.api_key = api_key
        else:
            config = get_config()
            self.api_key = config.get_runpod_api_key()
        
        if not self.api_key:
            raise ValueError(
                "RunPod API key not found. Configure it with:\n"
                "  elchat config --runpod-key YOUR_API_KEY\n"
                "Or set RUNPOD_API_KEY environment variable."
            )
        runpod.api_key = self.api_key
    
    def get_gpu_info(self, gpu_type: str) -> Optional[GPUInfo]:
        """Get detailed information about a GPU type."""
        try:
            info = runpod.get_gpu(gpu_type)
            return GPUInfo(
                id=info.get("id", gpu_type),
                name=info.get("displayName", gpu_type),
                vram_gb=info.get("memoryInGb", 0),
                community_price=info.get("communityPrice"),
                secure_price=info.get("securePrice"),
                spot_price=info.get("communitySpotPrice"),
                available=info.get("communityCloud", False) or info.get("secureCloud", False),
            )
        except Exception:
            return None
    
    def check_gpu_availability(self, gpu_type: str, cloud_type: str = "COMMUNITY") -> tuple[bool, Optional[float]]:
        """Check if a GPU is available and return its price.
        
        Returns:
            (available, price_per_hour)
        """
        info = self.get_gpu_info(gpu_type)
        if not info:
            return False, None
        
        if cloud_type == "COMMUNITY":
            return info.community_price is not None, info.community_price
        else:
            return info.secure_price is not None, info.secure_price
    
    def estimate_cost(self, gpu_type: str, gpu_count: int, hours: float, cloud_type: str = "COMMUNITY") -> Optional[float]:
        """Estimate the cost of running a job."""
        available, price = self.check_gpu_availability(gpu_type, cloud_type)
        if not available or price is None:
            return None
        return price * gpu_count * hours
    
    def validate_connection(self) -> bool:
        """Validate that the API key works."""
        try:
            runpod.get_gpus()
            return True
        except Exception:
            return False

    def create_pod(
        self,
        name: str,
        gpu_type: str,
        image: str = "runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
        volume_gb: int = 100,
        cloud_type: str = "SECURE",
        gpu_count: int = 1,
        env: Optional[dict] = None,
        docker_args: Optional[str] = None,
    ) -> PodInfo:
        """Create a new pod on RunPod.
        
        Args:
            name: Pod name
            gpu_type: GPU type ID (e.g., "NVIDIA RTX A6000")
            image: Docker image to use (default: RunPod pre-cached pytorch)
            volume_gb: Volume size in GB
            cloud_type: "SECURE" or "COMMUNITY"
            gpu_count: Number of GPUs (1, 2, 4, or 8)
            env: Environment variables dict (key: value)
            docker_args: Startup command/script to run
        """
        pod = runpod.create_pod(
            name=name,
            image_name=image,
            gpu_type_id=gpu_type,
            gpu_count=gpu_count,
            cloud_type=cloud_type,
            volume_in_gb=volume_gb,
            container_disk_in_gb=20,
            ports="22/tcp,8000/http",
            volume_mount_path="/workspace",
            env=env,
            docker_args=docker_args or "",
        )
        
        return PodInfo(
            id=pod["id"],
            name=name,
            status="PENDING",
            gpu_type=gpu_type,
            gpu_count=gpu_count,
        )

    def get_pod(self, pod_id: str) -> Optional[PodInfo]:
        """Get information about a pod."""
        pods = runpod.get_pods()
        for pod in pods:
            if pod["id"] == pod_id:
                ssh_info = self._extract_ssh_info(pod)
                return PodInfo(
                    id=pod["id"],
                    name=pod["name"],
                    status=pod["desiredStatus"],
                    gpu_type=pod.get("gpuType", "unknown"),
                    gpu_count=pod.get("gpuCount", 1),
                    ssh_host=ssh_info[0] if ssh_info else None,
                    ssh_port=ssh_info[1] if ssh_info else None,
                )
        return None

    def get_pod_by_name(self, name: str) -> Optional[PodInfo]:
        """Get a pod by its name."""
        pods = runpod.get_pods()
        for pod in pods:
            if pod["name"] == name:
                ssh_info = self._extract_ssh_info(pod)
                return PodInfo(
                    id=pod["id"],
                    name=pod["name"],
                    status=pod["desiredStatus"],
                    gpu_type=pod.get("gpuType", "unknown"),
                    gpu_count=pod.get("gpuCount", 1),
                    ssh_host=ssh_info[0] if ssh_info else None,
                    ssh_port=ssh_info[1] if ssh_info else None,
                )
        return None

    def wait_for_pod(self, pod_id: str, timeout: int = 300) -> PodInfo:
        """Wait for a pod to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            pod = self.get_pod(pod_id)
            if pod and pod.status == "RUNNING" and pod.ssh_host:
                return pod
            time.sleep(5)
        raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")

    def stop_pod(self, pod_id: str) -> bool:
        """Stop a pod."""
        try:
            runpod.stop_pod(pod_id)
            return True
        except Exception:
            return False

    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate (delete) a pod."""
        try:
            runpod.terminate_pod(pod_id)
            return True
        except Exception:
            return False

    def list_pods(self) -> list[PodInfo]:
        """List all pods."""
        pods = runpod.get_pods()
        result = []
        for pod in pods:
            ssh_info = self._extract_ssh_info(pod)
            result.append(PodInfo(
                id=pod["id"],
                name=pod["name"],
                status=pod["desiredStatus"],
                gpu_type=pod.get("gpuType", "unknown"),
                gpu_count=pod.get("gpuCount", 1),
                ssh_host=ssh_info[0] if ssh_info else None,
                ssh_port=ssh_info[1] if ssh_info else None,
            ))
        return result

    def _extract_ssh_info(self, pod: dict) -> Optional[tuple[str, int]]:
        """Extract SSH host and port from pod data."""
        runtime = pod.get("runtime")
        if not runtime:
            return None
        
        ports = runtime.get("ports", [])
        for port_info in ports:
            if port_info.get("privatePort") == 22:
                return (port_info.get("ip"), port_info.get("publicPort"))
        return None

