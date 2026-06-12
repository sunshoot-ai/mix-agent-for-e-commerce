# 系统架构设计

## 1. 架构目标

本项目是 Python-only 的多 Agent 电商推荐与营销系统。目标是用 Supervisor 编排用户画像、商品推荐、库存决策、营销文案、A/B 实验、反馈闭环和可观测能力，形成一条可运行、可评估、可监控的推荐链路。

## 2. 请求链路

```text
User Request
└─ FastAPI: python/main.py
   ├─ /api/v1/recommend
   │  └─ SupervisorOrchestrator
   │     ├─ Phase 1 并行
   │     │  ├─ UserProfileAgent -> FeatureStore -> UserProfile
   │     │  └─ ProductRecAgent -> candidate products
   │     ├─ Phase 2 并行
   │     │  ├─ ProductRecAgent -> LLM rerank
   │     │  └─ InventoryAgent -> available product ids + inventory alerts
   │     ├─ filter inventory
   │     ├─ MarketingCopyAgent -> personalized copy
   │     ├─ ABTestEngine -> experiment group / copy style
   │     └─ Observability -> traces / metrics / events
   ├─ /api/v1/recommend/graph
   │  └─ LangGraph StateGraph pipeline
   ├─ /api/v1/events
   │  └─ FeedbackService -> FeatureStore + A/B outcome + metrics
   └─ /api/v1/metrics / alerts / dashboard / simulations
```

## 3. Agent 职责

| Agent | 文件 | 职责 |
| --- | --- | --- |
| 用户画像 Agent | `python/agents/user_profile_agent.py` | 聚合实时行为、RFM、分群，生成用户画像 |
| 商品推荐 Agent | `python/agents/product_rec_agent.py` | 商品召回、排序、LLM 重排 |
| 库存决策 Agent | `python/agents/inventory_agent.py` | 库存过滤、库存状态、告警信号 |
| 营销文案 Agent | `python/agents/marketing_copy_agent.py` | 个性化文案生成、模板选择、合规过滤 |

## 4. 服务分层

```text
python/services/
├─ experimentation/
│  ├─ ab_test.py
│  ├─ feedback.py
│  └─ simulation.py
├─ features/
│  ├─ store.py
│  ├─ aggregation.py
│  ├─ feature_store.yaml
│  └─ feature_repo/
├─ operations/
│  ├─ metrics.py
│  ├─ inventory_alerts.py
│  └─ alerting.py
├─ observability/
│  ├─ tracing.py
│  ├─ events.py
│  ├─ metrics_registry.py
│  ├─ token_usage.py
│  └─ schemas.py
├─ reliability/
│  ├─ error_taxonomy.py
│  ├─ retry_policy.py
│  └─ fallback_handler.py
└─ user_profile/
   ├─ segment_selector.py
   └─ standards.py
```

## 5. 数据与反馈闭环

```text
Behavior Event
└─ /api/v1/events
   └─ FeedbackService
      ├─ FeatureStore.record_behavior()
      ├─ ABTestEngine.record_outcome()
      └─ MetricsCollector

Recommendation
└─ UserProfileAgent
   └─ FeatureStore.get_user_features()
      ├─ recent views / clicks / purchases
      ├─ preferred categories
      ├─ RFM score
      └─ offline tags
```

## 6. 特征系统

`FeatureStore` 和 `FeastFeatureStore` 当前共同位于 `python/services/features/store.py`。

| 类 | 作用 |
| --- | --- |
| `FeatureStore` | 主链路实时特征服务，支持 Redis 和内存 fallback，负责行为写入、窗口聚合、缓存和离线标签合并 |
| `FeastFeatureStore` | Feast 在线特征读取适配器，主要用于 Feast 驱动的 CTR 模拟和在线特征实验 |

保留两个类独立，不拆 `feast_store.py`，减少文件数量和 import 面。

## 7. 实验与评估

```text
python/services/experimentation/ab_test.py
├─ Hash bucket assignment
├─ Thompson Sampling
├─ segmented posterior
├─ public CTR simulation
└─ Feast CTR simulation

python/evaluation/
├─ workflow evaluation
├─ tool-call evaluation
├─ latency evaluation
├─ CTR simulation evaluation
└─ regression guardrails
```

模拟数据与结果统一放在：

```text
python/artifacts/simulations/
```

## 8. 可观测与运行监控

| 模块 | 职责 |
| --- | --- |
| `observability/tracing.py` | trace/span 上下文 |
| `observability/events.py` | event buffer |
| `observability/metrics_registry.py` | in-process metric registry |
| `operations/metrics.py` | agent 和业务指标聚合 |
| `operations/alerting.py` | 阈值告警、通知、dashboard |
| `operations/inventory_alerts.py` | 库存告警快照 |

## 9. 部署视图

```text
docker-compose.yml
├─ api    -> Python FastAPI container
├─ redis  -> behavior/features/cache
├─ milvus -> vector database
└─ mysql  -> business data
```

CI / demo 入口：

```text
python scripts/run_ctr_experiment.py
python -m services.features.aggregation
python scripts/prepare_feast.py
python scripts/run_ctr_experiment_feast.py
```

