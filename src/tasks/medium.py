"""Task 2 — Medium: Fix a bad join between orders and customers tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.generator import generate_medium_data


@dataclass
class MediumTask:
    """Two tables joined on the wrong key, producing nulls and dupes."""

    task_id: str = "medium"
    description: str = "Fix a bad join between orders and customers tables"
    max_steps: int = 15
    broken_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    expected_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    orders_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    customers_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.broken_df, self.expected_df = generate_medium_data()
        # Keep references to the source tables for fix_join
        self.orders_df = generate_medium_data.orders_df.copy()  # type: ignore[attr-defined]
        self.customers_df = generate_medium_data.customers_df.copy()  # type: ignore[attr-defined]

    def get_broken_df(self) -> pd.DataFrame:
        return self.broken_df.copy()

    def get_expected_df(self) -> pd.DataFrame:
        return self.expected_df.copy()
