from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation import (
    EvaluationCase,
    EvaluationSuite,
    Guardrail,
    LatencyEvaluator,
    RegressionGuardrails,
    ToolCallEvaluator,
    WorkflowEvaluator,
    WorkflowReplayRecord,
    evaluate_ctr_simulation_result,
    get_simulation_artifact,
    get_simulation_artifact_status,
    validate_ctr_artifact,
)
from services.observability import EventStatus, TraceEvent


def _event(
    name: str,
    *,
    component: str = "workflow",
    component_type: str = "workflow",
    status: EventStatus = EventStatus.OK,
    latency_ms: float = 0.0,
    tool_name: str = "",
    attributes: dict | None = None,
) -> TraceEvent:
    return TraceEvent(
        event_name=name,
        trace_id="trace-1",
        request_id="req-1",
        timestamp=1.0,
        component=component,
        component_type=component_type,
        status=status,
        latency_ms=latency_ms,
        tool_name=tool_name,
        attributes=attributes or {},
    )


def _case(events: list[TraceEvent], expected: dict | None = None) -> EvaluationCase:
    return EvaluationCase(
        name="happy_path",
        replay=WorkflowReplayRecord(
            trace_id="trace-1",
            request_id="req-1",
            workflow="supervisor_recommendation",
            events=events,
        ),
        expected=expected or {},
    )


def test_workflow_evaluator_passes_required_transitions():
    case = _case([
        _event("workflow.transition", attributes={"from": "start", "to": "phase1"}),
        _event("workflow.complete", latency_ms=100.0),
    ])
    result = WorkflowEvaluator(required_transitions=[("start", "phase1")]).evaluate(case)

    assert result.passed is True
    assert not result.failures


def test_tool_evaluator_flags_hallucinated_tool():
    case = _case([
        _event(
            "span.end",
            component="fake_tool",
            component_type="tool",
            tool_name="fake_tool",
        )
    ])
    result = ToolCallEvaluator(allowed_tools={"inventory_stock_lookup"}).evaluate(case)

    assert result.passed is False
    assert "hallucinated tool calls" in result.failures[0]


def test_latency_evaluator_applies_thresholds():
    case = _case([
        _event("span.end", component_type="workflow", latency_ms=900.0),
        _event("span.end", component_type="agent", latency_ms=120.0),
    ])
    result = LatencyEvaluator(
        workflow_latency_threshold_ms=500.0,
        agent_latency_threshold_ms=200.0,
    ).evaluate(case)

    assert result.passed is False
    assert "workflow p95 latency" in result.failures[0]


def test_evaluation_suite_and_guardrails():
    case = _case([
        _event("workflow.transition", attributes={"from": "start", "to": "phase1"}),
        _event("workflow.complete", latency_ms=100.0),
    ])
    suite = EvaluationSuite(
        "recommendation_replay",
        [WorkflowEvaluator(required_transitions=[("start", "phase1")])],
    )
    results = suite.run([case])
    guardrail = RegressionGuardrails([
        Guardrail("workflow_completion_rate", 1.0, ">="),
    ]).evaluate_results(results)

    assert results[0].passed is True
    assert guardrail.passed is True


def test_ctr_simulation_evaluation_summary():
    summary = evaluate_ctr_simulation_result(
        {
            "rounds": 100,
            "baseline": {"ctr": 0.05},
            "thompson": {"ctr": 0.06},
            "uplift": {
                "relative_percent": 20.0,
                "ctr_diff_95ci": [0.001, 0.02],
            },
            "posterior": {"treatment_llm": {"posterior_mean": 0.6}},
        },
        min_relative_uplift_percent=10.0,
        min_rounds=100,
        require_positive_ci=True,
    )

    assert summary["passed"] is True
    assert summary["metadata"]["posterior"]["treatment_llm"]["posterior_mean"] == 0.6


def test_ctr_simulation_evaluation_includes_multi_seed_metadata():
    summary = evaluate_ctr_simulation_result(
        {
            "rounds": 100,
            "baseline": {"ctr": 0.05},
            "thompson": {"ctr": 0.06},
            "uplift": {
                "relative_percent": 20.0,
                "ctr_diff_95ci": [0.001, 0.02],
            },
            "seed_count": 3,
            "multi_seed": {
                "mean_relative_uplift_percent": 18.0,
                "std_relative_uplift_percent": 1.0,
                "relative_uplift_percent_95ci": [16.0, 20.0],
            },
            "segments": {"high_intent": {"uplift": {"relative_percent": 10.0}}},
        },
        min_relative_uplift_percent=10.0,
        min_rounds=100,
    )

    assert summary["passed"] is True
    assert summary["metadata"]["multi_seed"]["mean_relative_uplift_percent"] == 18.0
    assert "high_intent" in summary["metadata"]["segments"]


def test_ctr_artifact_validation(tmp_path: Path):
    result_path = tmp_path / "ctr_simulation_result.json"
    evaluation_path = tmp_path / "ctr_simulation_result_evaluation.json"
    result_path.write_text(
        """
        {
          "rounds": 20000,
          "baseline": {"ctr": 0.05},
          "thompson": {"ctr": 0.06},
          "uplift": {"relative_percent": 20.0}
        }
        """,
        encoding="utf-8",
    )
    evaluation_path.write_text("""{"passed": true}""", encoding="utf-8")

    artifact = get_simulation_artifact("public", data_dir=tmp_path)
    validation = validate_ctr_artifact(
        artifact,
        min_relative_uplift_percent=10.0,
        min_rounds=20000,
    )

    assert artifact["available"] is True
    assert validation["passed"] is True


def test_ctr_artifact_status_marks_missing_files(tmp_path: Path):
    status = get_simulation_artifact_status(data_dir=tmp_path)

    assert status["public"]["available"] is False
    assert status["feast"]["available"] is False
