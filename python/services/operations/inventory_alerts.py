from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from services.observability import default_registry


@dataclass(slots=True)
class InventoryAlertEvent:
    alert_id: str
    level: str
    product_id: str
    name: str
    current_stock: int
    action: str
    request_id: str = ""
    scene: str = ""
    status: str = "active"
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "level": self.level,
            "product_id": self.product_id,
            "name": self.name,
            "current_stock": self.current_stock,
            "action": self.action,
            "request_id": self.request_id,
            "scene": self.scene,
            "status": self.status,
            "created_at": self.created_at,
        }


class InventoryAlertService:
    """Stores low-stock inventory alerts and emits metrics."""

    def __init__(self, max_events: int = 1000):
        self._events: deque[InventoryAlertEvent] = deque(maxlen=max_events)

    def record_alerts(
        self,
        low_stock_alerts: list[dict[str, Any]],
        *,
        request_id: str = "",
        scene: str = "",
    ) -> list[dict[str, Any]]:
        recorded = []
        for raw_alert in low_stock_alerts:
            event = self._build_event(raw_alert, request_id=request_id, scene=scene)
            self._events.append(event)
            recorded.append(event.to_dict())
            default_registry.inc(
                "inventory_alert_total",
                level=event.level,
                action=event.action,
                scene=scene,
            )
            default_registry.set_gauge(
                "inventory_latest_stock",
                float(event.current_stock),
                product_id=event.product_id,
                level=event.level,
            )
        return recorded

    def snapshot(self, limit: int = 100, level: str | None = None) -> list[dict[str, Any]]:
        events = list(self._events)
        if level:
            events = [event for event in events if event.level == level]
        return [event.to_dict() for event in events[-limit:]]

    def _build_event(
        self,
        raw_alert: dict[str, Any],
        *,
        request_id: str,
        scene: str,
    ) -> InventoryAlertEvent:
        product_id = str(raw_alert.get("product_id", "unknown"))
        level = str(raw_alert.get("level", "warning") or "warning")
        action = str(raw_alert.get("action", "plan_restock") or "plan_restock")
        current_stock = _safe_int(raw_alert.get("current_stock"))
        created_at = time.time()
        return InventoryAlertEvent(
            alert_id=f"inventory:{level}:{product_id}:{int(created_at)}",
            level=level,
            product_id=product_id,
            name=str(raw_alert.get("name", "")),
            current_stock=current_stock,
            action=action,
            request_id=request_id,
            scene=scene,
            created_at=created_at,
        )


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0
