"""
LangSmith Experiment Tracking Integration for Tygent.

This module logs DAG execution results to LangSmith for experiment
tracking and monitoring.
"""

from typing import Any, Dict, Optional, List

try:
    from langsmith import Client as LangSmithClient
except Exception:  # pragma: no cover - langsmith may not be installed
    LangSmithClient = None  # type: ignore


class LangSmithTracker:
    """Simple logger that sends DAG execution data to LangSmith."""

    def __init__(self, client: Any) -> None:
        if LangSmithClient is not None and isinstance(client, str):
            self.client = LangSmithClient(api_key=client)
        else:
            self.client = client

    async def log_run(
        self,
        dag_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> None:
        if not self.client:
            raise RuntimeError("LangSmith client is not configured")
        try:
            if hasattr(self.client, "create_run"):
                await self.client.create_run(
                    project_name=dag_name,
                    run_type="tygent_dag",
                    inputs=inputs,
                    outputs=outputs,
                    tags=tags or [],
                )
            elif hasattr(self.client, "log_run"):
                await self.client.log_run(
                    dag_name=dag_name,
                    inputs=inputs,
                    outputs=outputs,
                    tags=tags or [],
                )
        except Exception as e:  # pragma: no cover - runtime errors
            print(f"Failed to log execution to LangSmith: {e}")
            raise

