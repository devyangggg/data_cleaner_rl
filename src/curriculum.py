"""Curriculum manager for automatic task difficulty progression."""

from __future__ import annotations

from collections import deque
from typing import Deque


class CurriculumManager:
    def __init__(self) -> None:
        self.history: Deque[tuple[str, bool]] = deque(maxlen=10)
        self.current_level = "easy"

    def record(self, task_id: str, success: bool) -> None:
        self.history.append((task_id, success))
        self.current_level = self.next_task()

    def _success_rate(self, task_id: str) -> float:
        relevant = [ok for tid, ok in self.history if tid == task_id]
        if not relevant:
            return 0.0
        return sum(1 for ok in relevant if ok) / len(relevant)

    def next_task(self) -> str:
        easy_rate = self._success_rate("easy")
        medium_rate = self._success_rate("medium")

        if self.current_level == "easy":
            if easy_rate >= 0.8:
                return "medium"
            return "easy"

        if self.current_level == "medium":
            if medium_rate >= 0.7:
                return "hard"
            if easy_rate < 0.8:
                return "easy"
            return "medium"

        if medium_rate < 0.7:
            return "medium"
        return "hard"

    def get_stats(self) -> dict:
        return {
            "current_level": self.current_level,
            "recent_history": [
                {"task_id": task_id, "success": success}
                for task_id, success in self.history
            ],
            "easy_success_rate": round(self._success_rate("easy"), 3),
            "medium_success_rate": round(self._success_rate("medium"), 3),
        }
