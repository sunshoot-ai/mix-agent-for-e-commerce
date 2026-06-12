# 项目计划

## 1. 当前目标

项目当前目标是维护一个 Python-only 的多 Agent 电商推荐与营销系统。系统围绕 FastAPI、Supervisor 编排、LangGraph 展示链路、实时特征、A/B 实验、反馈闭环、评估和告警构建。

## 2. 当前范围

保留：

```text
python/
├─ main.py
├─ agents/
├─ config/
├─ models/
├─ orchestrator/
├─ services/
├─ evaluation/
├─ scripts/
├─ load_tests/
├─ artifacts/
└─ tests/
```

不在当前范围：

```text
非 Python 运行时实现
```

## 3. 核心模块

| 模块 | 职责 |
| --- | --- |
| `python/main.py` | FastAPI 入口，暴露推荐、反馈、实验、指标、告警和模拟 artifact API |
| `python/orchestrator/supervisor.py` | 生产推荐主编排，负责并行 Agent 调用、聚合、实验和观测 |
| `python/orchestrator/graph.py` | LangGraph 状态图推荐链路，用于展示和对比 |
| `python/agents/` | 四个核心 Agent 和 BaseAgent |
| `python/services/` | 运行时服务能力 |
| `python/evaluation/` | 离线评估和 guardrail |
| `python/scripts/` | CTR 实验、Feast 准备、行为生成脚本 |
| `python/artifacts/simulations/` | 样本数据和模拟结果 |

## 4. Services 目标结构

```text
services/
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
├─ reliability/
└─ user_profile/
```

## 5. 已完成重构

- 将实验能力收敛到 `services/experimentation/`。
- 将特征能力收敛到 `services/features/`。
- 将指标、库存告警、阈值告警收敛到 `services/operations/`。
- 将离线评估迁移到 `python/evaluation/`。
- 将脚本迁移到 `python/scripts/`。
- 将压测入口迁移到 `python/load_tests/`。
- 将模拟数据和结果迁移到 `python/artifacts/simulations/`。
- 将可观测指标注册表命名为 `metrics_registry.py`，避免和业务 metrics 混淆。

## 6. 运行计划

本地运行：

```bash
cd python
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Docker 运行：

```bash
docker compose up -d --build api redis milvus mysql
```

实验脚本：

```bash
cd python
python scripts/run_ctr_experiment.py
python -m services.features.aggregation
python scripts/prepare_feast.py
python scripts/run_ctr_experiment_feast.py
```

## 7. 验收计划

每次结构调整后运行：

```bash
python -m compileall python
python -m pytest python\tests
```

当前已验证：

```text
59 passed
```

## 8. 后续维护原则

- `services/` 只放运行时服务能力。
- 离线评估、脚本、产物不放在 `services/` 根目录。
- 新 Agent 放入 `python/agents/`，由 Supervisor 显式编排。
- 新实验能力优先放入 `services/experimentation/`。
- 新监控或告警能力优先放入 `services/operations/`。
- 新特征接入优先放入 `services/features/`。

