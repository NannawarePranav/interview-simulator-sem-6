"""Scoring and violation logging for interview monitoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class ViolationEvent:
    """Single violation entry."""

    timestamp: str
    violation_type: str
    message: str
    screenshot_path: str | None = None


@dataclass
class ScoringManager:
    """Tracks score and violation history."""

    initial_score: int = 10
    penalties: Dict[str, int] = field(
        default_factory=lambda: {
            "no_face": 2,
            "look_away": 1,
            "multiple_people": 1,
            "phone_detected": 1,
            "different_person": 2,
        }
    )
    score: int = field(init=False)
    events: List[ViolationEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.score = self.initial_score

    def add_violation(
        self, violation_type: str, message: str, screenshot_path: str | None = None
    ) -> None:
        penalty = self.penalties.get(violation_type, 1)
        self.score = max(0, self.score - penalty)
        event = ViolationEvent(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            violation_type=violation_type,
            message=message,
            screenshot_path=screenshot_path,
        )
        self.events.append(event)

    def get_violation_counts(self) -> Dict[str, int]:
        return dict(Counter(event.violation_type for event in self.events))

    def build_final_report(self) -> str:
        counts = self.get_violation_counts()
        lines = [
            "=" * 52,
            "Interview Monitoring Session Report",
            "=" * 52,
            f"Final Score: {self.score}/{self.initial_score}",
            f"Total Violations: {len(self.events)}",
            "",
            "Violation Breakdown:",
            f"- No face: {counts.get('no_face', 0)}",
            f"- Look away: {counts.get('look_away', 0)}",
            f"- Multiple people: {counts.get('multiple_people', 0)}",
            f"- Phone detected: {counts.get('phone_detected', 0)}",
            f"- Different person: {counts.get('different_person', 0)}",
            "",
            "Detailed Log:",
        ]
        if not self.events:
            lines.append("- No violations recorded.")
        else:
            for event in self.events:
                details = (
                    f"[{event.timestamp}] {event.violation_type}: {event.message}"
                )
                if event.screenshot_path:
                    details += f" (screenshot: {event.screenshot_path})"
                lines.append(f"- {details}")
        return "\n".join(lines)
