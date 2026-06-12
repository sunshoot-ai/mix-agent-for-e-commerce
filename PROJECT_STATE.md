# PROJECT STATE

## Goal
- Unify feature infrastructure under `python/services`, connect Feast-driven user features to main recommendation flow, and build reproducible CTR A/B simulation (baseline bucket split vs Thompson Sampling) targeting ~15% relative uplift in simulation.

## Completed
- Migrated feature-related assets from `python/agents` to `python/services` structure:
  - `feature_repo` moved/recreated under `python/services/features/feature_repo`
  - feature scripts moved to `python/services/features/aggregation.py`, `python/scripts/generate_behavior.py`
  - locust script moved to `python/load_tests/locustfile.py`
- Merged Feast client capability into `python/services/features/store.py`:
  - Added `FeastFeatureStore` with default repo path `python/services/features/feature_repo`
  - Removed hardcoded old repo path
  - Added lazy Feast import + graceful error handling
- Main flow rewired to services-layer feature access:
  - `python/main.py` imports `FeastFeatureStore` from `services.features.store`
  - `python/agents/user_profile_agent.py` imports from `services.features.store`
  - `UserProfileAgent` behavior collection now has fallback when Feast unavailable
- Extended A/B simulation framework in `python/services/experimentation/ab_test.py`:
  - Added public dataset loader + fallback generator
  - Added CTR simulation for baseline vs Thompson
  - Added Feast-driven simulation: `run_ctr_simulation_from_feast(...)`
  - Added uplift and 95% CI output
- Added runnable experiment scripts:
  - `python/scripts/run_ctr_experiment.py` (public dataset simulation)
  - `python/scripts/run_ctr_experiment_feast.py` (Feast-driven simulation)
- Produced result artifacts:
  - `python/artifacts/simulations/ctr_simulation_result.json`
  - `python/artifacts/simulations/ctr_simulation_feast_result.json`
  - `python/artifacts/simulations/ctr_simulation_feast_result_target15.json`
- Achieved target-like simulation uplift after tuning:
  - With `rounds=100000`, `treatment_effect=1.38`: relative uplift ≈ `15.27%`
- Added productionization infrastructure modules under `python/services`:
  - `observability/`: trace/span schemas, event buffer, metric registry, token usage extraction, trace context helpers
  - `reliability/`: retry policy, timeout enforcement, fallback metadata, structured error taxonomy
  - `evaluation/`: workflow replay evaluation, tool-call evaluation, latency evaluation, CTR simulation evaluation, regression guardrails
- Integrated observability/reliability into runtime paths:
  - `BaseAgent` now records reliability metadata, retry/fallback metrics, timeout status, and agent spans
  - Supervisor and LangGraph flows now emit workflow spans and transition events
  - LLM calls, Feast feature fetches, and inventory tool calls now emit spans/metrics
  - `/api/v1/metrics` includes observability snapshot and event count
- Extended experiment/evaluation integration:
  - `ABTestEngine.record_metric(...)` remains backward compatible and now accepts optional `segment`, `scene`, `policy`, `trace_id`
  - Added segmented metric aggregation via `get_segmented_stats(...)`
  - CTR scripts now write evaluation sidecar JSON files beside raw results
- Fixed Docker + Feast runtime path:
  - API container now mounts `./python:/app`, so Docker uses current workspace code
  - Feast Redis online store now points to Docker service `redis:6379`
  - Added `python/scripts/prepare_feast.py` to run Feast apply + materialize
  - `aggregate_features.py` and `generate_behavior.py` now respect `ECOM_REDIS_URL`
  - Regenerated Feast parquet and materialized 1000 rows into Redis online store
- Reduced demo log noise:
  - Observability events are still buffered/metricized, but per-event structlog output is disabled by default
  - Added `ECOM_OBSERVABILITY_LOG_EVENTS` and `ECOM_OBSERVABILITY_LOG_SAMPLE_RATE`
  - Feast CTR script suppresses repeated third-party Feast warnings for clean demo output
- Latest validation:
  - `GET /health` returns healthy from Docker API
  - `GET /features/1` returns Feast online features
  - Public CTR simulation evaluation passes
  - Feast CTR simulation evaluation passes with latest run: uplift ≈ `16.44%`, `evaluation_passed=True`
  - Targeted test suite passes: `20 passed`

## Current Files
- Core architecture/runtime:
  - `python/main.py`
  - `python/agents/user_profile_agent.py`
  - `python/services/__init__.py`
  - `python/services/features/store.py`
- Experiment/simulation:
  - `python/services/experimentation/ab_test.py`
  - `python/scripts/run_ctr_experiment.py`
  - `python/scripts/run_ctr_experiment_feast.py`
- Infrastructure:
  - `python/services/observability/`
  - `python/services/reliability/`
  - `python/evaluation/`
  - `python/scripts/prepare_feast.py`
- Feature repo and data:
  - `python/services/features/feature_repo/features.py`
  - `python/services/features/feature_repo/feature_store.yaml`
  - `python/services/features/aggregation.py`
  - `python/scripts/generate_behavior.py`
  - `python/artifacts/simulations/public_ecommerce_events_sample.csv`
  - `python/artifacts/simulations/ctr_simulation_feast_result_target15.json`
  - `python/artifacts/simulations/ctr_simulation_result_evaluation.json`
  - `python/artifacts/simulations/ctr_simulation_feast_result_target15_evaluation.json`

## Important Decisions
- Keep feature access in `services` layer, not in `agents`, to centralize data infrastructure responsibilities.
- Use `FeastFeatureStore` as online feature provider for user profiling and simulation prior construction.
- Preserve resilience:
  - Lazy Feast initialization
  - Fallback behavior in `UserProfileAgent` when Feast unavailable
- Define experiment comparison as:
  - baseline = `ABTestEngine.assign()` (static hash bucket split)
  - treatment policy = `ABTestEngine.assign_thompson()` + posterior updates
- Parameterize treatment effect (`treatment_effect`) to calibrate simulation uplift target.
- Keep observability low-coupling:
  - event buffer and metrics remain in-process
  - per-event logging is controlled by env vars and disabled by default for demo runs
- Run Feast-backed workflows inside Docker, not host Python, unless host environment has `feast` installed.
- Treat `python/services/features/feature_repo/feature_store.yaml` as the active Feast config; the old root-level services feature-store YAML is no longer used.
- Materialize Feast after regenerating parquet features:
  - `python services/features/aggregation.py`
  - `python scripts/prepare_feast.py`

## Failed Attempts
- Repeated sandbox/session command issues:
  - `CreateProcessAsUserW failed: 1312` for non-escalated shell operations
  - Required escalated runs for repository inspection/modification
- Cavecrew skill path blocked by platform risk gate:
  - `auto-review 503` prevented loading `.agents/skills/cavecrew/SKILL.md`
  - Work continued via approved non-cavecrew fallback implementation
- Initial migration command bug:
  - `Move-Item -LiteralPath ...\*` failed (wildcard with `LiteralPath`)
  - Led to partial move; feature repo had to be recreated safely in `services`
- Environment mismatch during early validation:
  - `ModuleNotFoundError: No module named 'feast'` when command used non-venv interpreter
  - Resolved by running checks with `.\.venv\Scripts\python.exe`
- Docker image/code mismatch:
  - API container initially had `feast==0.39.1` but stale code without `FeastFeatureStore`
  - Root cause: compose used image copy without source volume; opening Docker did not sync workspace code
  - Resolved by mounting `./python:/app` and recreating API container
- Feast Docker networking issue:
  - `feature_repo/feature_store.yaml` used `localhost:6379`, which points to the API container itself
  - Resolved by using `redis:6379`
- Feast materialization issue:
  - `prepare_feast.py` initially ran from `/app`, so `FileSource(path="data/user_behavior.parquet")` could not resolve
  - Resolved by changing working directory to the feature repo before `apply/materialize`
- Stale parquet timestamp issue:
  - Existing `user_behavior.parquet` was outside the materialization window, producing `0it`
  - Resolved by regenerating parquet via `aggregate_features.py`, then rematerializing 1000 rows
- Log flooding during Feast simulation:
  - Observability events initially emitted one structlog line per span
  - After disabling event logs, remaining flood came from Feast root warnings
  - Resolved with env-gated event logging and warning suppression in the Feast simulation script
- Experiment output inconsistency observed once:
  - console result and file snapshot differed temporarily due rerun/timing
  - resolved by rerun + explicit file timestamp/content verification

## Constraints
- Multi-agent e-commerce architecture with FastAPI entrypoint and async orchestration.
- Feast integration depends on local repo config and online store availability.
- Current Feast warnings exist (non-blocking), including:
  - feature view listing deprecation warnings
- Simulation uplift is synthetic/controlled; production uplift not guaranteed without online safeguards.
- Keep compatibility with existing `ABTestEngine` API and current schema contracts.
- Docker API service now relies on a source volume mount for local development. Rebuild is still required after dependency changes.
- `view_count_1h/24h/7d` may be zero if Redis behavior timestamps are outside the sliding windows; `avg_order_amount`, `rfm_score`, and `recent_views` can still be populated.

## Next Step
1. Make uplift estimation robust:
   - Run multi-seed evaluation (e.g., 20 seeds), store mean/std/CI.
2. Move from global to segmented Thompson:
   - Segment by scene/user cohort for better realism.
3. Add API access for simulation artifacts:
   - Endpoint to read latest public/Feast CTR simulation result and evaluation sidecar.
4. Add CI/demo guardrails:
   - Check script execution + expected metric schema + minimum uplift threshold in simulation.
5. Improve Feast feature realism:
   - Refresh/generate behavior events before aggregation so sliding-window view counts are non-zero.
