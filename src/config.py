"""Load explicit policy and cost assumptions from YAML."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.policy import CostConfig


@dataclass(frozen=True)
class PolicySettings:
    costs: CostConfig
    reviewers: int
    reviews_per_reviewer_per_day: int
    overflow_action: str

    @property
    def daily_review_capacity(self) -> int:
        return self.reviewers * self.reviews_per_reviewer_per_day

    def validate(self) -> None:
        self.costs.validate()
        for name, value in (
            ("reviewers", self.reviewers),
            ("reviews_per_reviewer_per_day", self.reviews_per_reviewer_per_day),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.overflow_action not in ("approve", "decline"):
            raise ValueError("overflow_action must be 'approve' or 'decline'.")


def load_policy_settings(path: str | Path = "configs/costs.yaml") -> PolicySettings:
    """Load YAML settings and validate all assumptions before use."""
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    cost_fields = set(CostConfig.__dataclass_fields__)
    costs = CostConfig(**{key: raw[key] for key in cost_fields if key in raw})
    settings = PolicySettings(
        costs=costs,
        reviewers=raw.get("reviewers", 0),
        reviews_per_reviewer_per_day=raw.get("reviews_per_reviewer_per_day", 0),
        overflow_action=raw.get("overflow_action", "approve"),
    )
    settings.validate()
    return settings
