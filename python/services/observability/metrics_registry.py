from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import quantiles
from threading import RLock
from time import time
from typing import Any

from .schemas import MetricSample


@dataclass(slots=True)
class MetricSeries:
    counter: float = 0.0
    gauge: float = 0.0
    samples: list[float] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)


class MetricRegistry:
    def __init__(self):
        self._series: dict[str, MetricSeries] = defaultdict(MetricSeries)
        self._samples: list[MetricSample] = []
        self._lock = RLock()

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            series = self._series[name]
            series.counter += value
            if labels:
                series.labels.update(labels)
            self._samples.append(
                MetricSample(name=name, value=value, labels=dict(labels), timestamp=time())
            )

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            series = self._series[name]
            series.gauge = value
            if labels:
                series.labels.update(labels)
            self._samples.append(
                MetricSample(name=name, value=value, labels=dict(labels), timestamp=time())
            )

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            series = self._series[name]
            series.samples.append(value)
            if labels:
                series.labels.update(labels)
            self._samples.append(
                MetricSample(name=name, value=value, labels=dict(labels), timestamp=time())
            )

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for name, series in self._series.items():
                result[name] = {
                    "counter": series.counter,
                    "gauge": series.gauge,
                    "count": len(series.samples),
                    "avg": sum(series.samples) / len(series.samples) if series.samples else 0.0,
                    "p50": self._percentile(series.samples, 0.50),
                    "p95": self._percentile(series.samples, 0.95),
                    "p99": self._percentile(series.samples, 0.99),
                    "labels": dict(series.labels),
                }
            return result

    def recent_samples(self, limit: int = 100) -> list[MetricSample]:
        with self._lock:
            return self._samples[-limit:]

    def export_prometheus_text(self) -> str:
        lines: list[str] = []
        snap = self.snapshot()
        for name, data in snap.items():
            lines.append(f"# HELP {name} Observability metric")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}_counter {data['counter']}")
            lines.append(f"{name}_gauge {data['gauge']}")
            if data["count"]:
                lines.append(f"{name}_count {data['count']}")
                lines.append(f"{name}_avg {data['avg']}")
                lines.append(f"{name}_p95 {data['p95']}")
        return "\n".join(lines)

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        try:
            return float(quantiles(values, n=100, method="inclusive")[int(quantile * 100) - 1])
        except Exception:
            ordered = sorted(values)
            index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
            return float(ordered[index])


default_registry = MetricRegistry()
