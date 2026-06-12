"""
Multi-Agent E-Commerce Recommendation System — FastAPI Entry Point

Endpoints:
  POST /api/v1/recommend          - 获取个性化推荐
  POST /api/v1/recommend/graph    - 通过LangGraph pipeline推荐
  GET  /api/v1/experiments        - 查看A/B实验状态
  GET  /api/v1/metrics            - 查看系统监控指标
  GET  /health                    - 健康检查
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from models.schemas import FeedbackRequest, FeedbackResponse, RecommendationRequest, RecommendationResponse
from orchestrator.supervisor import SupervisorOrchestrator
from orchestrator.graph import build_recommendation_graph
from services.experimentation.ab_test import ABTestEngine
from services.experimentation.feedback import FeedbackService
from services.features.store import FeatureStore
from services.operations.alerting import alert_service_from_settings
from services.operations.inventory_alerts import InventoryAlertService
from services.operations.metrics import MetricsCollector
from services.observability import default_registry, get_event_buffer
from evaluation import get_simulation_artifact, get_simulation_artifact_status

logger = structlog.get_logger()
settings = get_settings()


ab_engine = ABTestEngine()
metrics_collector = MetricsCollector()
realtime_feature_store = FeatureStore(ttl=settings.feature_ttl_seconds)
inventory_alert_service = InventoryAlertService()
alert_evaluation_service = alert_service_from_settings(
    settings,
    metric_registry=default_registry,
    inventory_alert_service=inventory_alert_service,
)
feedback_service = FeedbackService(
    feature_store=realtime_feature_store,
    ab_engine=ab_engine,
    metrics_collector=metrics_collector,
)
supervisor = SupervisorOrchestrator(
    ab_engine=ab_engine,
    feature_store=realtime_feature_store,
    inventory_alert_service=inventory_alert_service,
)
rec_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rec_graph
    rec_graph = build_recommendation_graph()
    logger.info("app.startup", model=settings.llm_model)
    yield
    logger.info("app.shutdown")


app = FastAPI(
    title="Multi-Agent E-Commerce Recommendation System",
    description="用户画像Agent + 商品推荐Agent + 营销文案Agent + 库存决策Agent，并行+聚合模式",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/features/{user_id}")
async def get_features(user_id: str):
    result = await realtime_feature_store.get_user_features(user_id)
    return JSONResponse(content=result, media_type="application/json")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": settings.llm_model,
        "simulation_artifacts": get_simulation_artifact_status(),
    }



@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """使用Supervisor编排器进行推荐 (生产推荐用法)"""
    response = await supervisor.recommend(request)
    _collect_metrics(response)
    return response


@app.post("/api/v1/events", response_model=FeedbackResponse)
async def record_behavior_event(request: FeedbackRequest):
    """记录曝光/点击/加购/购买等反馈事件，驱动特征与实验闭环"""
    return await feedback_service.record_event(request)


@app.post("/api/v1/recommend/graph")
async def recommend_via_graph(request: RecommendationRequest):
    """使用LangGraph状态图进行推荐 (展示LangGraph能力)"""
    if not rec_graph:
        return {"error": "Graph not initialized"}
    state = {
        "user_id": request.user_id,
        "scene": request.scene,
        "num_items": request.num_items,
        "context": request.context,
    }
    result = await rec_graph.ainvoke(state)
    return {
        "request_id": result.get("request_id"),
        "user_id": result.get("user_id"),
        "products": [p.model_dump() for p in result.get("final_products", [])],
        "marketing_copies": result.get("marketing_copies", []),
        "experiment_group": result.get("experiment_group", "control"),
        "total_latency_ms": round(result.get("total_latency_ms", 0), 1),
    }


@app.get("/api/v1/experiments")
async def get_experiments():
    """查看所有A/B实验状态"""
    experiments = {}
    for exp_id, exp in ab_engine.experiments.items():
        experiments[exp_id] = {
            "name": exp.name,
            "enabled": exp.enabled,
            "groups": [
                {
                    "name": g.name,
                    "weight": g.weight,
                    "config": g.config,
                    "successes": g.successes,
                    "failures": g.failures,
                }
                for g in exp.groups
            ],
            "stats": ab_engine.get_stats(exp_id),
        }
    return experiments


@app.get("/api/v1/metrics")
async def get_metrics():
    """查看系统监控指标"""
    return {
        "agents": metrics_collector.get_agent_stats(),
        "business": metrics_collector.get_business_stats(),
        "inventory_alerts": inventory_alert_service.snapshot(limit=20),
        "alerts": alert_evaluation_service.evaluate(),
        "observability": default_registry.snapshot(),
        "recent_event_count": len(get_event_buffer().snapshot()),
    }


@app.get("/api/v1/inventory/alerts")
async def get_inventory_alerts(limit: int = 100, level: str | None = None):
    """查看最近库存 warning/critical 告警"""
    return {
        "alerts": inventory_alert_service.snapshot(limit=limit, level=level),
    }


@app.get("/api/v1/alerts")
async def get_alerts():
    """查看阈值告警状态：耗时、fallback、Token、库存 critical。"""
    return alert_evaluation_service.evaluate()


@app.get("/api/v1/dashboard")
async def get_dashboard():
    """简易监控大盘聚合：告警、耗时、可靠性、成本、库存。"""
    return alert_evaluation_service.dashboard()


@app.get("/api/v1/alerts/history")
async def get_alert_history(limit: int = 100, status: str | None = None):
    """查看告警触发/恢复历史。"""
    return {
        "history": alert_evaluation_service.history(limit=limit, status=status),
    }


@app.get("/api/v1/alerts/notifications")
async def get_alert_notifications(limit: int = 100, status: str | None = None):
    """查看告警通知发送结果。"""
    return {
        "notifications": alert_evaluation_service.notifications(limit=limit, status=status),
    }


@app.get("/api/v1/simulations/ctr/{kind}")
async def get_ctr_simulation_artifact(kind: str):
    try:
        artifact = get_simulation_artifact(kind)
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={"error": "unknown_simulation_artifact", "kind": kind},
        )
    if not artifact["available"]:
        return JSONResponse(status_code=404, content=artifact)
    return artifact


@app.post("/api/v1/experiments/{experiment_id}/outcome")
async def record_outcome(experiment_id: str, group: str, success: bool):
    """记录A/B测试结果,更新Thompson Sampling"""
    ab_engine.record_outcome(experiment_id, group, success)
    return {"status": "recorded"}


def _collect_metrics(response: RecommendationResponse):
    for name, result in response.agent_results.items():
        metrics_collector.record_agent_call(
            agent_name=name,
            success=result.success,
            latency_ms=result.latency_ms,
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
