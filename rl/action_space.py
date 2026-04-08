"""Discrete action templates used by lightweight RL baselines."""

from __future__ import annotations

from typing import Any

EASY_ACTION_TEMPLATES: list[dict[str, Any]] = [
    {"command": "cast_column", "params": {"column": "revenue", "dtype": "float64"}},
    {"command": "cast_column", "params": {"column": "revenue", "dtype": "int64"}},
    {"command": "fix_date_format", "params": {"column": "date", "format": "%d-%m-%Y"}},
    {"command": "fix_date_format", "params": {"column": "date", "format": "%m-%d-%Y"}},
    {"command": "fill_nulls", "params": {"column": "units_sold", "value": 0}},
    {"command": "fill_nulls", "params": {"column": "units_sold", "value": -1}},
    {"command": "drop_duplicates", "params": {"subset": ["date"]}},
    {"command": "revert_step", "params": {}},
]
