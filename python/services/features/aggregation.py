import redis
import pandas as pd
import time
import os

from datetime import datetime, UTC

try:
    from scripts.generate_behavior import generate_events
except ImportError:
    from generate_behavior import generate_events

r = redis.from_url(
    os.getenv("ECOM_REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,
)

rows = []

now = int(time.time())

if os.getenv("ECOM_REFRESH_BEHAVIOR_BEFORE_AGGREGATE", "false").lower() == "true":
    generate_events(
        int(os.getenv("ECOM_REFRESH_BEHAVIOR_EVENT_COUNT", "5000")),
        max_event_age_seconds=int(os.getenv("ECOM_REFRESH_BEHAVIOR_MAX_AGE_SECONDS", "3600")),
        sleep_seconds=0.0,
    )
    now = int(time.time())

for user_id in range(1, 1001):

    key = f"user:{user_id}:views"

    # 1小时
    view_count_1h = r.zcount(
        key,
        now - 3600,
        now
    )

    # 24小时
    view_count_24h = r.zcount(
        key,
        now - 86400,
        now
    )

    # 7天
    view_count_7d = r.zcount(
        key,
        now - 604800,
        now
    )

    # 最近行为
    recent_views = r.zrevrange(
        key,
        0,
        10
    )

    # 用户金额
    profile = r.hgetall(
        f"user:{user_id}:profile"
    )

    avg_order_amount = float(
        profile.get(
            "last_order_amount",
            0
        )
    )

    # 简单RFM
    rfm_score = (
        view_count_7d * 0.3 +
        avg_order_amount * 0.7
    )

    rows.append({
        "user_id": str(user_id),

        "view_count_1h": view_count_1h,
        "view_count_24h": view_count_24h,
        "view_count_7d": view_count_7d,

        "avg_order_amount": avg_order_amount,

        "rfm_score": rfm_score,

        "recent_views": ",".join(recent_views),

        "event_timestamp": datetime.now(UTC)
    })

df = pd.DataFrame(rows)

# 获取当前脚本所在目录的绝对路径
base_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接成完整的输出路径
output_path = os.path.join(base_dir, "feature_repo", "data", "user_behavior.parquet")

# 自动创建目录
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df.to_parquet(output_path, index=False)

print("feature table generated")
