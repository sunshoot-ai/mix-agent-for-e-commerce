from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from config import Settings
from services.operations.inventory_alerts import InventoryAlertService
from services.observability import MetricRegistry, default_registry

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class AlertThreshold:
    metric: str
    field: str
    warning: float
    critical: float
    description: str


@dataclass(slots=True)
class AlertEvent:
    alert_id: str
    name: str
    level: str
    status: str
    metric: str
    field: str
    value: float
    threshold: float
    message: str
    labels: dict[str, str]
    triggered_at: float
    resolved_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "level": self.level,
            "status": self.status,
            "metric": self.metric,
            "field": self.field,
            "value": round(self.value, 4),
            "threshold": round(self.threshold, 4),
            "message": self.message,
            "labels": dict(self.labels),
            "triggered_at": self.triggered_at,
            "resolved_at": self.resolved_at,
        }


DEFAULT_THRESHOLDS: dict[str, AlertThreshold] = {
    "workflow_latency_p95": AlertThreshold(
        metric="workflow_total_latency_ms",
        field="p95",
        warning=3000.0,
        critical=5000.0,
        description="推荐主链路 P95 耗时过高",
    ),
    "agent_latency_p95": AlertThreshold(
        metric="agent_result_latency_ms",
        field="p95",
        warning=1500.0,
        critical=3000.0,
        description="Agent P95 耗时过高",
    ),
    "agent_fallback_total": AlertThreshold(
        metric="agent_fallback_total",
        field="counter",
        warning=1.0,
        critical=5.0,
        description="Agent fallback 次数异常",
    ),
    "llm_token_usage": AlertThreshold(
        metric="llm_total_tokens_total",
        field="counter",
        warning=8000.0,
        critical=20000.0,
        description="LLM Token 用量过高",
    ),
    "inventory_critical": AlertThreshold(
        metric="inventory_critical_active",
        field="gauge",
        warning=1.0,
        critical=3.0,
        description="库存 critical 告警未处理",
    ),
}


class AlertEvaluationService:
    """Evaluates observability metrics, keeps alert history, and emits notifications."""

    def __init__(
        self,
        metric_registry: MetricRegistry | None = None,
        inventory_alert_service: InventoryAlertService | None = None,
        thresholds: dict[str, AlertThreshold] | None = None,
        max_history: int = 500,
        webhook_url: str = "",
        webhook_timeout_seconds: float = 2.0,
    ):
        self.metric_registry = metric_registry or default_registry
        self.inventory_alert_service = inventory_alert_service
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self._alerts: dict[str, AlertEvent] = {}
        self._history: list[dict[str, Any]] = []
        self._max_history = max_history
        self.webhook_url = webhook_url
        self.webhook_timeout_seconds = webhook_timeout_seconds

    def evaluate(self) -> dict[str, Any]:
        metric_snapshot = self.metric_registry.snapshot()
        breached: dict[str, AlertEvent] = {}

        for name, threshold in self.thresholds.items():
            if name == "inventory_critical":
                event = self._evaluate_inventory_critical(name, threshold)
            else:
                event = self._evaluate_metric(name, threshold, metric_snapshot)
            if event:
                breached[event.alert_id] = event

        now = time.time()
        for alert_id, previous in list(self._alerts.items()):
            if alert_id not in breached and previous.status == "active":
                previous.status = "resolved"
                previous.resolved_at = now
                self._record_history(previous, "resolved")

        for alert_id, event in breached.items():
            current = self._alerts.get(alert_id)
            if current and current.status == "active":
                current.level = event.level
                current.value = event.value
                current.threshold = event.threshold
                current.message = event.message
                current.labels = event.labels
            else:
                self._alerts[alert_id] = event
                self._record_history(event, "triggered")

        active = [alert.to_dict() for alert in self._alerts.values() if alert.status == "active"]
        resolved = [
            alert.to_dict() for alert in self._alerts.values() if alert.status == "resolved"
        ]
        return {
            "active_count": len(active),
            "resolved_count": len(resolved),
            "active": sorted(active, key=lambda item: item["triggered_at"], reverse=True),
            "resolved": sorted(resolved, key=lambda item: item["resolved_at"] or 0, reverse=True),
            "thresholds": {
                name: {
                    "metric": threshold.metric,
                    "field": threshold.field,
                    "warning": threshold.warning,
                    "critical": threshold.critical,
                    "description": threshold.description,
                }
                for name, threshold in self.thresholds.items()
            },
            "history": self.history(limit=20),
        }

    def dashboard(self) -> dict[str, Any]:
        alert_snapshot = self.evaluate()
        metric_snapshot = self.metric_registry.snapshot()
        return {
            "alerts": {
                "active_count": alert_snapshot["active_count"],
                "resolved_count": alert_snapshot["resolved_count"],
                "active": alert_snapshot["active"],
            },
            "latency": {
                "workflow_p95_ms": _metric_value(
                    metric_snapshot, "workflow_total_latency_ms", "p95"
                ),
                "agent_p95_ms": _metric_value(metric_snapshot, "agent_result_latency_ms", "p95"),
            },
            "reliability": {
                "agent_fallback_total": _metric_value(
                    metric_snapshot, "agent_fallback_total", "counter"
                ),
                "workflow_completed_total": _metric_value(
                    metric_snapshot, "workflow_requests_total", "counter"
                ),
            },
            "cost": {
                "llm_total_tokens": _metric_value(
                    metric_snapshot, "llm_total_tokens_total", "counter"
                ),
            },
            "inventory": {
                "critical_active": len(self._inventory_critical_alerts()),
            },
            "history": self.history(limit=20),
        }

    def history(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        events = self._history
        if status:
            events = [event for event in events if event.get("status") == status]
        return list(events[-limit:])

    def notifications(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        events = [event for event in self._history if event.get("notification")]
        if status:
            events = [
                event
                for event in events
                if event.get("notification", {}).get("status") == status
            ]
        return list(events[-limit:])

    def _evaluate_metric(
        self,
        name: str,
        threshold: AlertThreshold,
        metric_snapshot: dict[str, dict[str, Any]],
    ) -> AlertEvent | None:
        if threshold.metric not in metric_snapshot:
            return None
        value = _metric_value(metric_snapshot, threshold.metric, threshold.field)
        return self._build_event_if_breached(name, threshold, value, metric_snapshot[threshold.metric].get("labels", {}))

    def _evaluate_inventory_critical(
        self,
        name: str,
        threshold: AlertThreshold,
    ) -> AlertEvent | None:
        value = float(len(self._inventory_critical_alerts()))
        return self._build_event_if_breached(name, threshold, value, {"source": "inventory_alerts"})

    def _build_event_if_breached(
        self,
        name: str,
        threshold: AlertThreshold,
        value: float,
        labels: dict[str, Any],
    ) -> AlertEvent | None:
        if value < threshold.warning:
            return None
        level = "critical" if value >= threshold.critical else "warning"
        active_threshold = threshold.critical if level == "critical" else threshold.warning
        return AlertEvent(
            alert_id=f"threshold:{name}",
            name=name,
            level=level,
            status="active",
            metric=threshold.metric,
            field=threshold.field,
            value=value,
            threshold=active_threshold,
            labels={str(key): str(val) for key, val in labels.items()},
            message=f"{threshold.description}: {value:.2f} >= {active_threshold:.2f}",
            triggered_at=time.time(),
        )

    def _record_history(self, alert: AlertEvent, event_type: str) -> None:
        item = alert.to_dict()
        item["event_type"] = event_type
        item["recorded_at"] = time.time()
        item["notification"] = self._send_notification(item)
        self._history.append(item)
        if len(self._history) > self._max_history:
            del self._history[: len(self._history) - self._max_history]
        default_registry.inc(
            "alert_events_total",
            alert=alert.name,
            level=alert.level,
            status=alert.status,
            event_type=event_type,
        )
        if event_type == "triggered":
            logger.warning(
                "alert.triggered",
                alert=alert.name,
                level=alert.level,
                value=alert.value,
                threshold=alert.threshold,
            )
        else:
            logger.info("alert.resolved", alert=alert.name, level=alert.level)

    def _send_notification(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.webhook_url:
            return {"status": "disabled"}
        try:
            response = httpx.post(
                self.webhook_url,
                json=payload,
                timeout=self.webhook_timeout_seconds,
            )
            success = 200 <= response.status_code < 300
            default_registry.inc(
                "alert_notifications_total",
                channel="webhook",
                status="success" if success else "failed",
            )
            return {
                "status": "success" if success else "failed",
                "channel": "webhook",
                "status_code": response.status_code,
            }
        except Exception as exc:
            default_registry.inc(
                "alert_notifications_total",
                channel="webhook",
                status="error",
            )
            logger.warning("alert.notification_failed", error=str(exc))
            return {
                "status": "error",
                "channel": "webhook",
                "error": str(exc),
            }

    def _inventory_critical_alerts(self) -> list[dict[str, Any]]:
        if not self.inventory_alert_service:
            return []
        return [
            alert
            for alert in self.inventory_alert_service.snapshot(limit=1000, level="critical")
            if alert.get("status", "active") == "active"
        ]


def _metric_value(
    metric_snapshot: dict[str, dict[str, Any]],
    metric: str,
    field: str,
) -> float:
    try:
        return float(metric_snapshot.get(metric, {}).get(field, 0.0) or 0.0)
    except Exception:
        return 0.0


def thresholds_from_settings(settings: Settings) -> dict[str, AlertThreshold]:
    return {
        "workflow_latency_p95": AlertThreshold(
            metric="workflow_total_latency_ms",
            field="p95",
            warning=settings.alert_workflow_latency_warning_ms,
            critical=settings.alert_workflow_latency_critical_ms,
            description="推荐主链路 P95 耗时过高",
        ),
        "agent_latency_p95": AlertThreshold(
            metric="agent_result_latency_ms",
            field="p95",
            warning=settings.alert_agent_latency_warning_ms,
            critical=settings.alert_agent_latency_critical_ms,
            description="Agent P95 耗时过高",
        ),
        "agent_fallback_total": AlertThreshold(
            metric="agent_fallback_total",
            field="counter",
            warning=settings.alert_agent_fallback_warning_count,
            critical=settings.alert_agent_fallback_critical_count,
            description="Agent fallback 次数异常",
        ),
        "llm_token_usage": AlertThreshold(
            metric="llm_total_tokens_total",
            field="counter",
            warning=settings.alert_llm_tokens_warning_count,
            critical=settings.alert_llm_tokens_critical_count,
            description="LLM Token 用量过高",
        ),
        "inventory_critical": AlertThreshold(
            metric="inventory_critical_active",
            field="gauge",
            warning=settings.alert_inventory_critical_warning_count,
            critical=settings.alert_inventory_critical_critical_count,
            description="库存 critical 告警未处理",
        ),
    }


def alert_service_from_settings(
    settings: Settings,
    *,
    metric_registry: MetricRegistry | None = None,
    inventory_alert_service: InventoryAlertService | None = None,
) -> AlertEvaluationService:
    return AlertEvaluationService(
        metric_registry=metric_registry,
        inventory_alert_service=inventory_alert_service,
        thresholds=thresholds_from_settings(settings),
        webhook_url=settings.alert_webhook_url,
        webhook_timeout_seconds=settings.alert_webhook_timeout_seconds,
    )
