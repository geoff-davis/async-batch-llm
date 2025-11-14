"""Metrics collection observer."""

import asyncio
import json
from typing import Any

from .base import BaseObserver, ProcessingEvent


class MetricsObserver(BaseObserver):
    """Collect metrics for monitoring (thread-safe)."""

    def __init__(self, *, max_processing_samples: int = 100):
        """Initialize metrics collector."""
        if max_processing_samples <= 0:
            raise ValueError("max_processing_samples must be positive")
        self.metrics: dict[str, Any] = {
            "items_processed": 0,
            "items_succeeded": 0,
            "items_failed": 0,
            "rate_limits_hit": 0,
            "total_cooldown_time": 0.0,
            "processing_times": [],
            "error_counts": {},
            "processing_times_count": 0,
            "processing_times_sum": 0.0,
        }
        self._processing_times: list[float] = []
        self._max_processing_samples = max_processing_samples
        self._lock = asyncio.Lock()

    async def on_event(
        self,
        event: ProcessingEvent,
        data: dict[str, Any],
    ) -> None:
        """Collect metrics from events (thread-safe)."""
        async with self._lock:
            if event == ProcessingEvent.ITEM_COMPLETED:
                self.metrics["items_processed"] += 1
                self.metrics["items_succeeded"] += 1
                if "duration" in data:
                    duration = float(data["duration"])
                    self.metrics["processing_times_sum"] += duration
                    self.metrics["processing_times_count"] += 1
                    self._processing_times.append(duration)
                    if len(self._processing_times) > self._max_processing_samples:
                        self._processing_times.pop(0)

            elif event == ProcessingEvent.ITEM_FAILED:
                self.metrics["items_processed"] += 1
                self.metrics["items_failed"] += 1
                if "error_type" in data:
                    error_type = data["error_type"]
                    self.metrics["error_counts"][error_type] = (
                        self.metrics["error_counts"].get(error_type, 0) + 1
                    )

            elif event == ProcessingEvent.RATE_LIMIT_HIT:
                self.metrics["rate_limits_hit"] += 1

            elif event == ProcessingEvent.COOLDOWN_ENDED:
                if "duration" in data:
                    self.metrics["total_cooldown_time"] += data["duration"]

    async def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics with computed statistics (thread-safe)."""
        async with self._lock:
            return {
                **{k: v for k, v in self.metrics.items() if k != "processing_times"},
                "processing_times": list(self._processing_times),
                "avg_processing_time": (

                        self.metrics["processing_times_sum"] / self.metrics["processing_times_count"]
                        if self.metrics["processing_times_count"] > 0
                        else 0

                ),
                "success_rate": (
                    self.metrics["items_succeeded"] / self.metrics["items_processed"]
                    if self.metrics["items_processed"] > 0
                    else 0
                ),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            "items_processed": 0,
            "items_succeeded": 0,
            "items_failed": 0,
            "rate_limits_hit": 0,
            "total_cooldown_time": 0.0,
            "processing_times": [],
            "error_counts": {},
            "processing_times_count": 0,
            "processing_times_sum": 0.0,
        }
        self._processing_times = []

    async def export_json(self) -> str:
        """Export metrics as JSON string.

        Returns:
            JSON string containing all metrics and computed statistics

        Example:
            >>> observer = MetricsObserver()
            >>> # ... process items ...
            >>> json_str = await observer.export_json()
            >>> print(json_str)
        """
        metrics = await self.get_metrics()
        # Convert processing_times list to just count for cleaner export
        export_data = {
            **{k: v for k, v in metrics.items() if k != "processing_times"},
            "processing_times_count": metrics.get("processing_times_count", 0),
        }
        return json.dumps(export_data, indent=2)

    async def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string

        Example:
            >>> observer = MetricsObserver()
            >>> # ... process items ...
            >>> prom_text = await observer.export_prometheus()
            >>> print(prom_text)
            # HELP batch_llm_items_processed Total items processed
            # TYPE batch_llm_items_processed counter
            batch_llm_items_processed 100
            ...
        """
        metrics = await self.get_metrics()

        lines = []

        # Counter metrics
        counters = [
            ("items_processed", "Total items processed"),
            ("items_succeeded", "Total items succeeded"),
            ("items_failed", "Total items failed"),
            ("rate_limits_hit", "Total rate limits encountered"),
        ]

        for metric_name, help_text in counters:
            lines.append(f"# HELP batch_llm_{metric_name} {help_text}")
            lines.append(f"# TYPE batch_llm_{metric_name} counter")
            lines.append(f"batch_llm_{metric_name} {metrics.get(metric_name, 0)}")
            lines.append("")

        # Gauge metrics
        gauges = [
            ("avg_processing_time", "Average processing time in seconds"),
            ("success_rate", "Success rate (0.0 to 1.0)"),
            ("total_cooldown_time", "Total time spent in rate limit cooldown (seconds)"),
            ("processing_times_count", "Number of recorded processing time samples"),
        ]

        for metric_name, help_text in gauges:
            lines.append(f"# HELP batch_llm_{metric_name} {help_text}")
            lines.append(f"# TYPE batch_llm_{metric_name} gauge")
            lines.append(f"batch_llm_{metric_name} {metrics.get(metric_name, 0)}")
            lines.append("")

        # Error counts as labeled counter
        error_counts = metrics.get("error_counts", {})
        if error_counts:
            lines.append("# HELP batch_llm_errors_total Total errors by type")
            lines.append("# TYPE batch_llm_errors_total counter")
            for error_type, count in error_counts.items():
                # Sanitize error type for Prometheus label
                safe_type = error_type.replace('"', '\\"')
                lines.append(f'batch_llm_errors_total{{error_type="{safe_type}"}} {count}')
            lines.append("")

        return "\n".join(lines)

    async def export_dict(self) -> dict[str, Any]:
        """Export metrics as a dictionary.

        Returns:
            Dictionary containing all metrics and computed statistics

        Example:
            >>> observer = MetricsObserver()
            >>> # ... process items ...
            >>> data = await observer.export_dict()
            >>> print(data["success_rate"])
        """
        return await self.get_metrics()
