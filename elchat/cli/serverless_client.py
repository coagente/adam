"""
RunPod Serverless Client.

Handles communication with RunPod Serverless endpoints for training jobs.
No SSH required - everything via HTTP API.
"""

import os
import time
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class JobStatus:
    """Status of a serverless job."""
    id: str
    status: str  # IN_QUEUE, IN_PROGRESS, COMPLETED, FAILED, CANCELLED
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


class ServerlessClient:
    """Client for RunPod Serverless API."""
    
    BASE_URL = "https://api.runpod.ai/v2"
    
    def __init__(self, api_key: Optional[str] = None, endpoint_id: Optional[str] = None):
        """
        Initialize the serverless client.
        
        Args:
            api_key: RunPod API key. If not provided, reads from config.
            endpoint_id: Serverless endpoint ID.
        """
        if api_key:
            self.api_key = api_key
        else:
            from elchat.cli.config import get_config
            config = get_config()
            self.api_key = config.get_runpod_api_key()
        
        if not self.api_key:
            raise ValueError("RunPod API key not configured. Use: elchat config set --runpod-key YOUR_KEY")
        
        self.endpoint_id = endpoint_id
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def set_endpoint(self, endpoint_id: str):
        """Set the endpoint ID."""
        self.endpoint_id = endpoint_id
    
    def _get_endpoint_url(self, path: str) -> str:
        """Get full URL for endpoint."""
        if not self.endpoint_id:
            raise ValueError("Endpoint ID not set")
        return f"{self.BASE_URL}/{self.endpoint_id}/{path}"
    
    def submit_job(self, input_data: Dict[str, Any]) -> str:
        """
        Submit a training job to the serverless endpoint.
        
        Args:
            input_data: Job input parameters
            
        Returns:
            Job ID
        """
        url = self._get_endpoint_url("run")
        
        payload = {"input": input_data}
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data["id"]
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get the status of a job.
        
        Args:
            job_id: The job ID returned from submit_job
            
        Returns:
            JobStatus with current status and output
        """
        url = self._get_endpoint_url(f"status/{job_id}")
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        data = response.json()
        
        return JobStatus(
            id=data.get("id", job_id),
            status=data.get("status", "UNKNOWN"),
            output=data.get("output"),
            error=data.get("error"),
            execution_time=data.get("executionTime"),
        )
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: The job ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        url = self._get_endpoint_url(f"cancel/{job_id}")
        
        response = requests.post(url, headers=self.headers)
        return response.status_code == 200
    
    def wait_for_job(
        self,
        job_id: str,
        poll_interval: int = 10,
        timeout: int = 0,
        progress_callback=None,
    ) -> JobStatus:
        """
        Wait for a job to complete, polling status periodically.
        
        Args:
            job_id: The job ID to wait for
            poll_interval: Seconds between status checks
            timeout: Max seconds to wait (0 = no limit)
            progress_callback: Function to call with status updates
            
        Returns:
            Final JobStatus
        """
        start_time = time.time()
        
        while True:
            status = self.get_job_status(job_id)
            
            if progress_callback:
                progress_callback(status)
            
            if status.status in ["COMPLETED", "FAILED", "CANCELLED"]:
                return status
            
            if timeout > 0 and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")
            
            time.sleep(poll_interval)
    
    def run_sync(
        self,
        input_data: Dict[str, Any],
        poll_interval: int = 10,
        timeout: int = 0,
        progress_callback=None,
    ) -> JobStatus:
        """
        Submit a job and wait for it to complete.
        
        Convenience method that combines submit_job and wait_for_job.
        
        Args:
            input_data: Job input parameters
            poll_interval: Seconds between status checks
            timeout: Max seconds to wait (0 = no limit)
            progress_callback: Function to call with status updates
            
        Returns:
            Final JobStatus
        """
        job_id = self.submit_job(input_data)
        return self.wait_for_job(job_id, poll_interval, timeout, progress_callback)
    
    def health_check(self) -> bool:
        """
        Check if the endpoint is healthy.
        
        Returns:
            True if endpoint is responding
        """
        try:
            url = self._get_endpoint_url("health")
            response = requests.get(url, headers=self.headers, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_endpoint_info(self) -> Dict[str, Any]:
        """
        Get information about the endpoint.
        
        Returns:
            Endpoint configuration and status
        """
        # Note: This requires the management API, not the inference API
        url = f"https://api.runpod.io/graphql"
        
        query = """
        query getEndpoint($id: String!) {
            myself {
                serverlessDiscount
                endpoints(input: {id: $id}) {
                    id
                    name
                    gpuIds
                    workersMin
                    workersMax
                    idleTimeout
                }
            }
        }
        """
        
        response = requests.post(
            url,
            headers=self.headers,
            json={"query": query, "variables": {"id": self.endpoint_id}},
        )
        
        if response.status_code == 200:
            data = response.json()
            endpoints = data.get("data", {}).get("myself", {}).get("endpoints", [])
            if endpoints:
                return endpoints[0]
        
        return {}

