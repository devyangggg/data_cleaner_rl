"""Task 1 — Easy: Fix type, null, and date bugs in a sales dataset."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.generator import generate_easy_data


@dataclass
class EasyTask:
    """Sales data with 3 bugs: string revenue, wrong date format, NaN units."""

    task_id: str = "easy"
    description: str = "Fix type and null bugs in a sales dataset"
    max_steps: int = 10
    broken_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    expected_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.broken_df, self.expected_df = generate_easy_data()

    def get_broken_df(self) -> pd.DataFrame:
        """Return a fresh copy of the broken dataframe."""
        return self.broken_df.copy()

    def get_expected_df(self) -> pd.DataFrame:
        """Return a fresh copy of the expected dataframe."""
        return self.expected_df.copy()
