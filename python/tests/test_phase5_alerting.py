from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from config.settings import Settings
from main import app
from services.operations.alerting import (
    AlertEvaluationService,
    alert_service_from_settings,
    thresholds_from_settings,
)
from services.operations.inventory_alerts import InventoryAlertService
from services.observability import MetricRegistry


def test_alert_service_triggers_and_resolves_latency_alert():
    registry = MetricRegistry()
    service = AlertEvaluationService(metric_registry=registry)

    registry.observe("workflow_total_latency_ms", 6200.0, workflow="supervisor")
    snapshot = service.evaluate()

    assert snapshot["active_count"] == 1
    assert snapshot["active"][0]["name"] == "workflow_latency_p95"
    assert snapshot["active"][0]["level"] == "critical"

    clean_service = AlertEvaluationService(metric_registry=MetricRegistry())
    assert clean_service.evaluate()["active_count"] == 0


def test_alert_service_triggers_inventory_critical_alert():
    inventory_alerts = InventoryAlertService()
    inventory_alerts.record_alerts(
        [
            {
                "product_id": "P015",
                "name": "极低库存商品",
                "current_stock": 10,
                "level": "critical",
                "action": "urgent_restock",
            }
        ],
        request_id="req-1",
        scene="homepage",
    )
    service = AlertEvaluationService(
        metric_registry=MetricRegistry(),
        inventory_alert_service=inventory_alerts,
    )

    snapshot = service.evaluate()

    assert snapshot["active_count"] == 1
    assert snapshot["active"][0]["name"] == "inventory_critical"
    assert snapshot["active"][0]["value"] == 1.0


def test_alert_and_dashboard_api_expose_alert_state():
    client = TestClient(app)

    alerts = client.get("/api/v1/alerts")
    dashboard = client.get("/api/v1/dashboard")

    assert alerts.status_code == 200
    assert "active" in alerts.json()
    assert dashboard.status_code == 200
    assert "latency" in dashboard.json()
    assert "reliability" in dashboard.json()


def test_alert_service_records_history_and_resolution():
    registry = MetricRegistry()
    service = AlertEvaluationService(metric_registry=registry)

    registry.inc("agent_fallback_total", 1, agent="user_profile")
    triggered = service.evaluate()
    assert triggered["active_count"] == 1
    assert service.history(limit=10)[0]["event_type"] == "triggered"

    service.metric_registry = MetricRegistry()
    resolved = service.evaluate()
    assert resolved["active_count"] == 0
    assert resolved["resolved_count"] == 1
    assert service.history(limit=10)[-1]["event_type"] == "resolved"


def test_thresholds_can_be_built_from_settings():
    settings = Settings(
        alert_workflow_latency_warning_ms=111.0,
        alert_workflow_latency_critical_ms=222.0,
    )
    thresholds = thresholds_from_settings(settings)

    assert thresholds["workflow_latency_p95"].warning == 111.0
    assert thresholds["workflow_latency_p95"].critical == 222.0


def test_alert_history_api_exposes_events():
    client = TestClient(app)

    response = client.get("/api/v1/alerts/history")

    assert response.status_code == 200
    assert "history" in response.json()


def test_alert_notifications_disabled_by_default():
    registry = MetricRegistry()
    service = AlertEvaluationService(metric_registry=registry)

    registry.observe("workflow_total_latency_ms", 6200.0)
    service.evaluate()
    notifications = service.notifications()

    assert notifications[0]["notification"]["status"] == "disabled"


def test_alert_service_from_settings_wires_webhook_config():
    settings = Settings(
        alert_webhook_url="https://alerts.example.test/hook",
        alert_webhook_timeout_seconds=1.25,
    )
    service = alert_service_from_settings(settings, metric_registry=MetricRegistry())

    assert service.webhook_url == "https://alerts.example.test/hook"
    assert service.webhook_timeout_seconds == 1.25


def test_alert_notifications_api_exposes_delivery_state():
    client = TestClient(app)

    response = client.get("/api/v1/alerts/notifications")

    assert response.status_code == 200
    assert "notifications" in response.json()
