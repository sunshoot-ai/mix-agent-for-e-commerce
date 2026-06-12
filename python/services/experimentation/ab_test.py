"""
A/B测试引擎
- 流量分桶：用户ID哈希取模分桶
- 实验层：Agent级别 / 模型级别 / Prompt级别实验
- MAB算法：Thompson Sampling动态分配流量
- 指标收集：CTR / CVR / GMV / 停留时长
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Experiment:
    id: str
    name: str
    groups: list[ExperimentGroup]
    enabled: bool = True
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class ExperimentGroup:
    name: str
    weight: int = 50
    config: dict[str, Any] = field(default_factory=dict)
    # Thompson Sampling state
    successes: int = 1
    failures: int = 1


class ABTestEngine:
    """Bucket-based A/B test engine with optional Thompson Sampling."""

    def __init__(self, bucket_count: int = 100):
        self.bucket_count = bucket_count
        self.experiments: dict[str, Experiment] = {}
        self._metrics: list[dict[str, Any]] = []
        self._segment_posteriors: dict[tuple[str, str], list[ExperimentGroup]] = {}
        self._init_default_experiments()

    def _init_default_experiments(self):
        self.register_experiment(
            Experiment(
                id="rec_strategy",
                name="推荐策略实验",
                groups=[
                    ExperimentGroup(name="control", weight=50, config={"rerank": "rule_based"}),
                    ExperimentGroup(name="treatment_llm", weight=50, config={"rerank": "llm"}),
                ],
            )
        )
        self.register_experiment(
            Experiment(
                id="copy_style",
                name="文案风格实验",
                groups=[
                    ExperimentGroup(name="formal", weight=50, config={"style": "formal"}),
                    ExperimentGroup(name="casual", weight=50, config={"style": "casual"}),
                ],
            )
        )

    def register_experiment(self, exp: Experiment):
        self.experiments[exp.id] = exp

    def assign(self, user_id: str, experiment_id: str = "rec_strategy") -> dict[str, Any]:
        """Assign user to an experiment group using consistent hashing."""
        exp = self.experiments.get(experiment_id)
        if not exp or not exp.enabled:
            return {"group": "control", "config": {}}

        bucket = self._hash_bucket(user_id, experiment_id)
        group = self._bucket_to_group(bucket, exp.groups)
        return {"group": group.name, "config": group.config}

    def assign_thompson(self, user_id: str, experiment_id: str = "rec_strategy") -> dict[str, Any]:
        """Use Thompson Sampling for dynamic traffic allocation."""
        exp = self.experiments.get(experiment_id)
        if not exp or not exp.enabled:
            return {"group": "control", "config": {}}

        samples = []
        for g in exp.groups:
            sample = np.random.beta(g.successes, g.failures)
            samples.append((sample, g))

        best = max(samples, key=lambda x: x[0])[1]
        return {"group": best.name, "config": best.config}

    def assign_thompson_segmented(
        self,
        user_id: str,
        experiment_id: str = "rec_strategy",
        segment: str = "global",
    ) -> dict[str, Any]:
        """Use an independent Thompson posterior for a scene/cohort segment."""
        exp = self.experiments.get(experiment_id)
        if not exp or not exp.enabled:
            return {"group": "control", "config": {}, "segment": segment}

        groups = self._get_segment_groups(exp, segment)
        samples = [(np.random.beta(g.successes, g.failures), g) for g in groups]
        best = max(samples, key=lambda x: x[0])[1]
        return {"group": best.name, "config": best.config, "segment": segment}

    def record_outcome(self, experiment_id: str, group_name: str, success: bool):
        """Update Thompson Sampling posterior with observed outcome."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return
        for g in exp.groups:
            if g.name == group_name:
                if success:
                    g.successes += 1
                else:
                    g.failures += 1
                break

    def record_segment_outcome(
        self,
        experiment_id: str,
        group_name: str,
        success: bool,
        segment: str = "global",
    ):
        """Update a segment-local Thompson posterior without touching global state."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return
        for g in self._get_segment_groups(exp, segment):
            if g.name == group_name:
                if success:
                    g.successes += 1
                else:
                    g.failures += 1
                break

    def get_segment_posteriors(self, experiment_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for (exp_id, segment), groups in self._segment_posteriors.items():
            if exp_id != experiment_id:
                continue
            result[segment] = {
                g.name: {
                    "successes": g.successes,
                    "failures": g.failures,
                    "posterior_mean": g.successes / (g.successes + g.failures),
                }
                for g in groups
            }
        return result

    def record_metric(
        self,
        experiment_id: str,
        group_name: str,
        metric_name: str,
        value: float,
        user_id: str = "",
        segment: str = "",
        scene: str = "",
        policy: str = "",
        trace_id: str = "",
    ):
        self._metrics.append({
            "experiment_id": experiment_id,
            "group": group_name,
            "metric": metric_name,
            "value": value,
            "user_id": user_id,
            "segment": segment,
            "scene": scene,
            "policy": policy,
            "trace_id": trace_id,
            "timestamp": time.time(),
        })

    def get_stats(self, experiment_id: str) -> dict[str, Any]:
        """Aggregate metrics per group for a given experiment."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return {}
        relevant = [m for m in self._metrics if m["experiment_id"] == experiment_id]
        stats: dict[str, dict[str, list[float]]] = {}
        for m in relevant:
            grp = m["group"]
            metric = m["metric"]
            if grp not in stats:
                stats[grp] = {}
            if metric not in stats[grp]:
                stats[grp][metric] = []
            stats[grp][metric].append(m["value"])

        result: dict[str, Any] = {}
        for grp, metrics in stats.items():
            result[grp] = {}
            for metric_name, values in metrics.items():
                arr = np.array(values)
                result[grp][metric_name] = {
                    "count": len(values),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
        return result

    def get_segmented_stats(
        self,
        experiment_id: str,
        *,
        segment_key: str = "segment",
    ) -> dict[str, Any]:
        """Aggregate experiment metrics by an optional recorded segment field."""
        relevant = [
            m for m in self._metrics
            if m["experiment_id"] == experiment_id and m.get(segment_key)
        ]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for metric in relevant:
            grouped.setdefault(str(metric.get(segment_key, "unknown")), []).append(metric)

        result: dict[str, Any] = {}
        for segment, metrics in grouped.items():
            by_group: dict[str, dict[str, list[float]]] = {}
            for metric in metrics:
                group = metric["group"]
                metric_name = metric["metric"]
                by_group.setdefault(group, {}).setdefault(metric_name, []).append(metric["value"])

            result[segment] = {}
            for group, group_metrics in by_group.items():
                result[segment][group] = {}
                for metric_name, values in group_metrics.items():
                    arr = np.array(values)
                    result[segment][group][metric_name] = {
                        "count": len(values),
                        "mean": float(arr.mean()),
                        "std": float(arr.std()),
                        "min": float(arr.min()),
                        "max": float(arr.max()),
                    }
        return result

    def _hash_bucket(self, user_id: str, experiment_id: str) -> int:
        raw = f"{user_id}:{experiment_id}"
        h = hashlib.md5(raw.encode()).hexdigest()
        return int(h[:8], 16) % self.bucket_count

    def _bucket_to_group(
        self, bucket: int, groups: list[ExperimentGroup]
    ) -> ExperimentGroup:
        total_weight = sum(g.weight for g in groups)
        cumulative = 0
        normalized_bucket = bucket * total_weight / self.bucket_count
        for g in groups:
            cumulative += g.weight
            if normalized_bucket < cumulative:
                return g
        return groups[-1]

    def _get_segment_groups(self, exp: Experiment, segment: str) -> list[ExperimentGroup]:
        key = (exp.id, segment or "global")
        if key not in self._segment_posteriors:
            self._segment_posteriors[key] = [
                ExperimentGroup(name=g.name, weight=g.weight, config=dict(g.config))
                for g in exp.groups
            ]
        return self._segment_posteriors[key]


def load_public_ecommerce_dataset(
    local_csv_path: str = "python/artifacts/simulations/public_ecommerce_events_sample.csv",
) -> pd.DataFrame:
    """
    Load public e-commerce behavior dataset.
    Priority:
    1) Local cached/checked-in CSV sample (public schema-compatible)
    2) Remote public UCI Online Retail dataset (if network available)
    3) Synthetic fallback with public-like schema
    """
    try:
        df = pd.read_csv(local_csv_path)
        if "user_id" in df.columns and "event_type" in df.columns:
            return df
    except Exception:
        pass

    remote_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    )
    try:
        raw = pd.read_excel(remote_url)
        raw = raw.rename(
            columns={
                "CustomerID": "user_id",
                "StockCode": "product_id",
                "Quantity": "quantity",
                "UnitPrice": "price",
            }
        )
        raw = raw.dropna(subset=["user_id"])
        raw["user_id"] = raw["user_id"].astype(str)
        raw["event_type"] = np.where(raw["quantity"] > 0, "purchase", "return")
        raw["price"] = raw["price"].fillna(0.0).astype(float)
        return raw[["user_id", "product_id", "event_type", "price"]]
    except Exception:
        return _generate_public_like_fallback()


def _generate_public_like_fallback(rows: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_ids = rng.integers(1, 5001, size=rows).astype(str)
    product_ids = rng.integers(1, 1001, size=rows)
    prices = rng.lognormal(mean=4.2, sigma=0.7, size=rows)
    event_probs = rng.uniform(0, 1, size=rows)
    event_type = np.where(
        event_probs < 0.10,
        "purchase",
        np.where(event_probs < 0.45, "click", "view"),
    )
    return pd.DataFrame(
        {
            "user_id": user_ids,
            "product_id": [f"P{pid:04d}" for pid in product_ids],
            "event_type": event_type,
            "price": prices.astype(float),
        }
    )


def run_ctr_simulation(
    events: pd.DataFrame,
    rounds: int = 15000,
    seed: int = 7,
    treatment_effect: float = 1.36,
    segmented: bool = False,
) -> dict[str, Any]:
    """
    Compare baseline bucket assignment vs Thompson Sampling.
    Baseline: ABTestEngine.assign (static 50/50 bucket split)
    Treatment selector: ABTestEngine.assign_thompson (adaptive MAB)
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    ab_baseline = ABTestEngine()
    ab_thompson = ABTestEngine()

    # Calibrate user propensity from public behavior events
    user_ctr_prior = _build_user_ctr_prior(events)
    all_users = user_ctr_prior["user_id"].to_numpy()
    all_priors = user_ctr_prior["prior_ctr"].to_numpy()
    all_segments = user_ctr_prior["segment"].to_numpy()

    baseline_clicks = 0
    thompson_clicks = 0
    baseline_impressions = 0
    thompson_impressions = 0
    segment_counters: dict[str, dict[str, dict[str, int]]] = {}

    # arm lift assumptions for online experiment simulation
    # control = rule_based, treatment_llm = better expected CTR
    arm_effect = {
        "control": 1.00,
        "treatment_llm": treatment_effect,
    }

    for _ in range(rounds):
        idx = int(rng.integers(0, len(all_users)))
        user_id = str(all_users[idx])
        user_base_p = float(all_priors[idx])
        segment = str(all_segments[idx])

        baseline_group = ab_baseline.assign(user_id)["group"]
        p_base = _clip_prob(user_base_p * arm_effect.get(baseline_group, 1.0))
        click_base = bool(rng.uniform() < p_base)
        baseline_clicks += int(click_base)
        baseline_impressions += 1

        if segmented:
            th_assignment = ab_thompson.assign_thompson_segmented(user_id, segment=segment)
        else:
            th_assignment = ab_thompson.assign_thompson(user_id)
        th_group = th_assignment["group"]
        p_th = _clip_prob(user_base_p * arm_effect.get(th_group, 1.0))
        click_th = bool(rng.uniform() < p_th)
        thompson_clicks += int(click_th)
        thompson_impressions += 1
        if segmented:
            ab_thompson.record_segment_outcome("rec_strategy", th_group, click_th, segment)
            _record_segment_counter(segment_counters, segment, "baseline", click_base)
            _record_segment_counter(segment_counters, segment, "thompson", click_th)
        else:
            ab_thompson.record_outcome("rec_strategy", th_group, click_th)

    baseline_ctr = baseline_clicks / baseline_impressions if baseline_impressions else 0.0
    thompson_ctr = thompson_clicks / thompson_impressions if thompson_impressions else 0.0
    uplift_abs = thompson_ctr - baseline_ctr
    uplift_rel = uplift_abs / baseline_ctr if baseline_ctr > 0 else 0.0

    ci_low, ci_high = _normal_approx_ci_diff(
        thompson_clicks,
        thompson_impressions,
        baseline_clicks,
        baseline_impressions,
    )

    result = {
        "rounds": rounds,
        "segmented": segmented,
        "baseline": {
            "impressions": baseline_impressions,
            "clicks": baseline_clicks,
            "ctr": baseline_ctr,
        },
        "thompson": {
            "impressions": thompson_impressions,
            "clicks": thompson_clicks,
            "ctr": thompson_ctr,
        },
        "uplift": {
            "absolute": uplift_abs,
            "relative": uplift_rel,
            "relative_percent": uplift_rel * 100.0,
            "ctr_diff_95ci": [ci_low, ci_high],
        },
        "posterior": (
            ab_thompson.get_segment_posteriors("rec_strategy")
            if segmented
            else _extract_group_posteriors(ab_thompson, "rec_strategy")
        ),
    }
    if segmented:
        result["segments"] = _summarize_segment_counters(segment_counters)
    return result


def run_ctr_simulation_multi_seed(
    events: pd.DataFrame,
    seeds: list[int] | None = None,
    rounds: int = 15000,
    treatment_effect: float = 1.36,
    segmented: bool = False,
) -> dict[str, Any]:
    seeds = seeds or list(range(20))
    per_seed = [
        run_ctr_simulation(
            events=events,
            rounds=rounds,
            seed=seed,
            treatment_effect=treatment_effect,
            segmented=segmented,
        )
        for seed in seeds
    ]
    return _summarize_multi_seed(per_seed, seeds, source="public_dataset")


async def run_ctr_simulation_from_feast(
    user_ids: list[str] | None = None,
    rounds: int = 15000,
    seed: int = 7,
    treatment_effect: float = 1.36,
    segmented: bool = False,
) -> dict[str, Any]:
    """
    CTR simulation driven by Feast online features.
    """
    from services.features.store import FeastFeatureStore

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    ab_baseline = ABTestEngine()
    ab_thompson = ABTestEngine()

    if not user_ids:
        user_ids = [str(i) for i in range(1, 1001)]

    feast = FeastFeatureStore()
    user_priors = await _build_user_ctr_prior_from_feast(feast, user_ids)
    all_users = user_priors["user_id"].to_numpy()
    all_priors = user_priors["prior_ctr"].to_numpy()
    all_segments = user_priors["segment"].to_numpy()

    baseline_clicks = 0
    thompson_clicks = 0
    baseline_impressions = 0
    thompson_impressions = 0
    segment_counters: dict[str, dict[str, dict[str, int]]] = {}

    arm_effect = {
        "control": 1.00,
        "treatment_llm": treatment_effect,
    }

    for _ in range(rounds):
        idx = int(rng.integers(0, len(all_users)))
        user_id = str(all_users[idx])
        user_base_p = float(all_priors[idx])
        segment = str(all_segments[idx])

        baseline_group = ab_baseline.assign(user_id)["group"]
        p_base = _clip_prob(user_base_p * arm_effect.get(baseline_group, 1.0))
        click_base = bool(rng.uniform() < p_base)
        baseline_clicks += int(click_base)
        baseline_impressions += 1

        if segmented:
            th_assignment = ab_thompson.assign_thompson_segmented(user_id, segment=segment)
        else:
            th_assignment = ab_thompson.assign_thompson(user_id)
        th_group = th_assignment["group"]
        p_th = _clip_prob(user_base_p * arm_effect.get(th_group, 1.0))
        click_th = bool(rng.uniform() < p_th)
        thompson_clicks += int(click_th)
        thompson_impressions += 1
        if segmented:
            ab_thompson.record_segment_outcome("rec_strategy", th_group, click_th, segment)
            _record_segment_counter(segment_counters, segment, "baseline", click_base)
            _record_segment_counter(segment_counters, segment, "thompson", click_th)
        else:
            ab_thompson.record_outcome("rec_strategy", th_group, click_th)

    baseline_ctr = baseline_clicks / baseline_impressions if baseline_impressions else 0.0
    thompson_ctr = thompson_clicks / thompson_impressions if thompson_impressions else 0.0
    uplift_abs = thompson_ctr - baseline_ctr
    uplift_rel = uplift_abs / baseline_ctr if baseline_ctr > 0 else 0.0
    ci_low, ci_high = _normal_approx_ci_diff(
        thompson_clicks,
        thompson_impressions,
        baseline_clicks,
        baseline_impressions,
    )
    result = {
        "source": "feast_online_features",
        "rounds": rounds,
        "segmented": segmented,
        "user_count": len(user_ids),
        "baseline": {
            "impressions": baseline_impressions,
            "clicks": baseline_clicks,
            "ctr": baseline_ctr,
        },
        "thompson": {
            "impressions": thompson_impressions,
            "clicks": thompson_clicks,
            "ctr": thompson_ctr,
        },
        "uplift": {
            "absolute": uplift_abs,
            "relative": uplift_rel,
            "relative_percent": uplift_rel * 100.0,
            "ctr_diff_95ci": [ci_low, ci_high],
        },
        "posterior": (
            ab_thompson.get_segment_posteriors("rec_strategy")
            if segmented
            else _extract_group_posteriors(ab_thompson, "rec_strategy")
        ),
    }
    if segmented:
        result["segments"] = _summarize_segment_counters(segment_counters)
    return result


async def run_ctr_simulation_from_feast_multi_seed(
    user_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    rounds: int = 15000,
    treatment_effect: float = 1.36,
    segmented: bool = False,
) -> dict[str, Any]:
    seeds = seeds or list(range(20))
    per_seed = []
    for seed in seeds:
        per_seed.append(
            await run_ctr_simulation_from_feast(
                user_ids=user_ids,
                rounds=rounds,
                seed=seed,
                treatment_effect=treatment_effect,
                segmented=segmented,
            )
        )
    return _summarize_multi_seed(per_seed, seeds, source="feast_online_features")


async def _build_user_ctr_prior_from_feast(
    feast_store: Any,
    user_ids: list[str],
) -> pd.DataFrame:
    rows = []
    for uid in user_ids:
        try:
            feat = await feast_store.get_user_features(uid)
        except Exception:
            feat = {}
        rows.append(
            {
                "user_id": str(uid),
                "view_count_1h": float(feat.get("view_count_1h", 0) or 0),
                "view_count_24h": float(feat.get("view_count_24h", 0) or 0),
                "view_count_7d": float(feat.get("view_count_7d", 0) or 0),
                "avg_order_amount": float(feat.get("avg_order_amount", 0) or 0),
                "rfm_score": float(feat.get("rfm_score", 0) or 0),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            {"user_id": ["cold_start"], "prior_ctr": [0.05], "segment": ["cold_start"]}
        )

    df["ctr_raw"] = (
        0.02
        + 0.08 * np.tanh(df["view_count_1h"] / 20.0)
        + 0.06 * np.tanh(df["view_count_24h"] / 60.0)
        + 0.04 * np.tanh(df["view_count_7d"] / 200.0)
        + 0.05 * np.tanh(df["rfm_score"] / 1000.0)
        + 0.03 * np.tanh(df["avg_order_amount"] / 5000.0)
    )
    df["prior_ctr"] = df["ctr_raw"].clip(0.01, 0.30)
    df["segment"] = np.where(
        df["view_count_24h"] > 0,
        "active_viewer",
        np.where(df["rfm_score"] >= 1000.0, "high_value", "standard"),
    )
    return df[["user_id", "prior_ctr", "segment"]]


def _build_user_ctr_prior(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    if "event_type" not in df.columns:
        df["event_type"] = "view"
    df["is_click"] = (df["event_type"] == "click").astype(int)
    grp = (
        df.groupby("user_id", as_index=False)
        .agg(clicks=("is_click", "sum"), total=("is_click", "count"))
    )
    grp["prior_ctr"] = (grp["clicks"] + 1) / (grp["total"] + 20)  # smoothed Beta prior
    grp["prior_ctr"] = grp["prior_ctr"].clip(0.01, 0.25)
    grp["segment"] = np.where(
        grp["prior_ctr"] >= 0.08,
        "high_intent",
        np.where(grp["total"] >= 5, "known_user", "light_user"),
    )
    if grp.empty:
        grp = pd.DataFrame(
            {"user_id": ["cold_start"], "prior_ctr": [0.05], "segment": ["cold_start"]}
        )
    return grp[["user_id", "prior_ctr", "segment"]]


def _record_segment_counter(
    counters: dict[str, dict[str, dict[str, int]]],
    segment: str,
    policy: str,
    clicked: bool,
) -> None:
    bucket = counters.setdefault(
        segment,
        {
            "baseline": {"impressions": 0, "clicks": 0},
            "thompson": {"impressions": 0, "clicks": 0},
        },
    )[policy]
    bucket["impressions"] += 1
    bucket["clicks"] += int(clicked)


def _summarize_segment_counters(
    counters: dict[str, dict[str, dict[str, int]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for segment, policies in counters.items():
        baseline = policies["baseline"]
        thompson = policies["thompson"]
        baseline_ctr = baseline["clicks"] / baseline["impressions"] if baseline["impressions"] else 0.0
        thompson_ctr = thompson["clicks"] / thompson["impressions"] if thompson["impressions"] else 0.0
        uplift_abs = thompson_ctr - baseline_ctr
        uplift_rel = uplift_abs / baseline_ctr if baseline_ctr > 0 else 0.0
        summary[segment] = {
            "baseline": {**baseline, "ctr": baseline_ctr},
            "thompson": {**thompson, "ctr": thompson_ctr},
            "uplift": {
                "absolute": uplift_abs,
                "relative": uplift_rel,
                "relative_percent": uplift_rel * 100.0,
            },
        }
    return summary


def _summarize_multi_seed(
    per_seed: list[dict[str, Any]],
    seeds: list[int],
    *,
    source: str,
) -> dict[str, Any]:
    uplifts = np.array([
        float(result.get("uplift", {}).get("relative_percent", 0.0) or 0.0)
        for result in per_seed
    ])
    mean = float(uplifts.mean()) if len(uplifts) else 0.0
    std = float(uplifts.std(ddof=1)) if len(uplifts) > 1 else 0.0
    margin = float(1.96 * std / np.sqrt(len(uplifts))) if len(uplifts) > 1 else 0.0
    first = per_seed[0] if per_seed else {}
    compact_per_seed = []
    for seed, result in zip(seeds, per_seed):
        compact_per_seed.append({
            "seed": seed,
            "baseline_ctr": result.get("baseline", {}).get("ctr", 0.0),
            "thompson_ctr": result.get("thompson", {}).get("ctr", 0.0),
            "relative_uplift_percent": result.get("uplift", {}).get("relative_percent", 0.0),
            "segment_uplift_percent": {
                segment: values.get("uplift", {}).get("relative_percent", 0.0)
                for segment, values in result.get("segments", {}).items()
            },
        })
    return {
        "source": source,
        "rounds": int(first.get("rounds", 0) or 0),
        "segmented": bool(first.get("segmented", False)),
        "seed_count": len(seeds),
        "seeds": seeds,
        "multi_seed": {
            "mean_relative_uplift_percent": mean,
            "std_relative_uplift_percent": std,
            "relative_uplift_percent_95ci": [mean - margin, mean + margin],
        },
        "per_seed": compact_per_seed,
    }


def _clip_prob(v: float) -> float:
    return float(min(0.95, max(0.001, v)))


def _normal_approx_ci_diff(
    success_a: int,
    n_a: int,
    success_b: int,
    n_b: int,
    z: float = 1.96,
) -> tuple[float, float]:
    p_a = success_a / n_a if n_a else 0.0
    p_b = success_b / n_b if n_b else 0.0
    se = np.sqrt((p_a * (1 - p_a) / max(n_a, 1)) + (p_b * (1 - p_b) / max(n_b, 1)))
    diff = p_a - p_b
    return float(diff - z * se), float(diff + z * se)


def _extract_group_posteriors(ab: ABTestEngine, experiment_id: str) -> dict[str, Any]:
    exp = ab.experiments.get(experiment_id)
    if not exp:
        return {}
    out = {}
    for g in exp.groups:
        out[g.name] = {
            "successes": g.successes,
            "failures": g.failures,
            "posterior_mean": g.successes / (g.successes + g.failures),
        }
    return out
