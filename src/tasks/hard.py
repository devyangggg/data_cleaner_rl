"""Task 3 — Hard: Trace and fix silent value corruption in a revenue pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.generator import generate_hard_data


@dataclass
class HardTask:
    """Exchange rate applied twice to USD rows — values look plausible."""

    task_id: str = "hard"
    description: str = "Trace and fix silent value corruption in a revenue pipeline"
    max_steps: int = 20
    broken_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    expected_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.broken_df, self.expected_df = generate_hard_data()

    def get_broken_df(self) -> pd.DataFrame:
        return self.broken_df.copy()

    def get_expected_df(self) -> pd.DataFrame:
        return self.expected_df.copy()
