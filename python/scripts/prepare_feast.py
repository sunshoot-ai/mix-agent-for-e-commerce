from __future__ import annotations

import sys
import os
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feast import FeatureStore
from services.features.feature_repo.features import user, user_behavior_fv


def main() -> None:
    repo_path = Path(__file__).resolve().parents[1] / "services" / "features" / "feature_repo"
    os.chdir(repo_path)
    store = FeatureStore(repo_path=str(repo_path))
    store.apply([user, user_behavior_fv])
    store.materialize_incremental(end_date=datetime.now(UTC))
    print("feast_repo", repo_path)
    print("feast_materialized_until", datetime.now(UTC).isoformat())


if __name__ == "__main__":
    main()
