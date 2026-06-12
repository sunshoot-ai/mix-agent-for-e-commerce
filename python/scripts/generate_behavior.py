import os
import random
import time

import redis

r = redis.from_url(
    os.getenv("ECOM_REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,
)

products = [
    "iPhone",
    "MacBook",
    "iPad",
    "AirPods",
    "显示器",
    "机械键盘",
]

def generate_events(
    event_count: int | None = None,
    *,
    max_event_age_seconds: int = 0,
    sleep_seconds: float = 0.01,
) -> int:
    generated = 0
    while event_count is None or generated < event_count:
        user_id = random.randint(1, 1000)
        product = random.choice(products)
        now = int(time.time())
        age = random.randint(0, max_event_age_seconds) if max_event_age_seconds > 0 else 0
        ts = now - age

        r.zadd(f"user:{user_id}:views", {f"{product}_{ts}_{generated}": ts})

        order_amount = random.randint(50, 5000)
        r.hset(
            f"user:{user_id}:profile",
            mapping={"last_order_amount": order_amount},
        )

        generated += 1
        print(f"user={user_id}, product={product}, ts={ts}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return generated


def main() -> None:
    raw_count = os.getenv("ECOM_BEHAVIOR_EVENT_COUNT")
    event_count = int(raw_count) if raw_count else None
    max_age = int(os.getenv("ECOM_BEHAVIOR_MAX_EVENT_AGE_SECONDS", "0"))
    sleep_seconds = float(os.getenv("ECOM_BEHAVIOR_SLEEP_SECONDS", "0.01"))
    generate_events(
        event_count,
        max_event_age_seconds=max_age,
        sleep_seconds=sleep_seconds,
    )


if __name__ == "__main__":
    main()
