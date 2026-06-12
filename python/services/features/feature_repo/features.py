from datetime import timedelta

from feast import (
    Entity,
    FeatureView,
    Field,
)

from feast.types import (
    Int64,
    Float32,
    String,
)

from feast.infra.offline_stores.file_source import FileSource

user = Entity(
    name="user_id",
    join_keys=["user_id"],
)

user_behavior_source = FileSource(
    path="data/user_behavior.parquet",
    timestamp_field="event_timestamp",
)

user_behavior_fv = FeatureView(
    name="user_behavior_features",

    entities=[user],

    ttl=timedelta(days=7),

    schema=[

        Field(
            name="view_count_1h",
            dtype=Int64
        ),

        Field(
            name="view_count_24h",
            dtype=Int64
        ),

        Field(
            name="view_count_7d",
            dtype=Int64
        ),

        Field(
            name="avg_order_amount",
            dtype=Float32
        ),

        Field(
            name="rfm_score",
            dtype=Float32
        ),

        Field(
            name="recent_views",
            dtype=String
        ),
    ],

    source=user_behavior_source,

    online=True,
)
