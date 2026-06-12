from __future__ import annotations

from typing import Any

UNKNOWN = "unknown"

VALID_GENDERS = {"male", "female", UNKNOWN}
VALID_AGE_GROUPS = {UNKNOWN, "18-24", "25-34", "35-44", "45-54", "55+"}

GENDER_ALIASES = {
    "m": "male",
    "man": "male",
    "male": "male",
    "boy": "male",
    "男": "male",
    "男性": "male",
    "男生": "male",
    "先生": "male",
    "f": "female",
    "woman": "female",
    "female": "female",
    "girl": "female",
    "女": "female",
    "女性": "female",
    "女生": "female",
    "女士": "female",
    "unknown": UNKNOWN,
    "unk": UNKNOWN,
    "未知": UNKNOWN,
    "保密": UNKNOWN,
    "其他": UNKNOWN,
}

CITY_ALIASES = {
    "beijing": "北京",
    "bj": "北京",
    "北京": "北京",
    "北京市": "北京",
    "shanghai": "上海",
    "sh": "上海",
    "上海": "上海",
    "上海市": "上海",
    "guangzhou": "广州",
    "gz": "广州",
    "广州": "广州",
    "广州市": "广州",
    "shenzhen": "深圳",
    "sz": "深圳",
    "深圳": "深圳",
    "深圳市": "深圳",
    "hangzhou": "杭州",
    "hz": "杭州",
    "杭州": "杭州",
    "杭州市": "杭州",
    "chengdu": "成都",
    "cd": "成都",
    "成都": "成都",
    "成都市": "成都",
    "wuhan": "武汉",
    "wh": "武汉",
    "武汉": "武汉",
    "武汉市": "武汉",
    "nanjing": "南京",
    "nj": "南京",
    "南京": "南京",
    "南京市": "南京",
    "xian": "西安",
    "xi'an": "西安",
    "西安": "西安",
    "西安市": "西安",
    "suzhou": "苏州",
    "苏州": "苏州",
    "苏州市": "苏州",
    "unknown": UNKNOWN,
    "未知": UNKNOWN,
}


def normalize_gender(value: Any) -> str:
    """Normalize gender to male/female/unknown without raising exceptions."""
    try:
        if value is None:
            return UNKNOWN
        text = str(value).strip().lower()
        if not text:
            return UNKNOWN
        normalized = GENDER_ALIASES.get(text, UNKNOWN)
        return normalized if normalized in VALID_GENDERS else UNKNOWN
    except Exception:
        return UNKNOWN


def normalize_age(value: Any) -> int | None:
    """Normalize age to an integer in [0, 120], otherwise None."""
    try:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            number = float(text)
        else:
            number = float(value)
        if not number.is_integer():
            return None
        age = int(number)
        if 0 <= age <= 120:
            return age
        return None
    except Exception:
        return None


def normalize_age_group(value: Any) -> str:
    """Map age to a standard age group."""
    age = normalize_age(value)
    if age is None or age < 18:
        return UNKNOWN
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    return "55+"


def normalize_city(value: Any) -> str:
    """Normalize city to a Chinese city name, otherwise unknown."""
    try:
        if value is None:
            return UNKNOWN
        text = str(value).strip()
        if not text:
            return UNKNOWN
        lookup = text.lower()
        if lookup in CITY_ALIASES:
            return CITY_ALIASES[lookup]
        if text.endswith("市") and len(text) > 1:
            text = text[:-1]
        if _looks_like_chinese_city(text):
            return text
        return UNKNOWN
    except Exception:
        return UNKNOWN


def build_standard_profile_fields(
    *,
    age: Any = None,
    gender: Any = None,
    city: Any = None,
) -> dict[str, Any]:
    """Build standardized profile fields for tags/analytics without changing raw fields."""
    normalized_age = normalize_age(age)
    normalized_gender = normalize_gender(gender)
    normalized_city = normalize_city(city)
    age_group = normalize_age_group(normalized_age)
    return {
        "gender": normalized_gender,
        "age": normalized_age,
        "age_group": age_group if age_group in VALID_AGE_GROUPS else UNKNOWN,
        "city": normalized_city,
    }


def _looks_like_chinese_city(value: str) -> bool:
    if not value or len(value) > 12:
        return False
    return all("\u4e00" <= char <= "\u9fff" or char in "·-" for char in value)
