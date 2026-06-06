from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class SessionState:
    candidate_profile: str = ""
    topics_to_cover: List[str] = field(default_factory=list)
    topics_covered: List[str] = field(default_factory=list)
    questions_asked: List[Dict[str, Any]] = field(default_factory=list)
    scores: Dict[str, List[float]] = field(default_factory=dict)
    current_topic: str = ""
    consecutive_weak_answers: int = 0
    proctor_score: int = 10
    proctor_violations: int = 0
    proctor_violations_breakdown: Dict[str, int] = field(default_factory=dict)
    proctor_violations_log: List[Dict[str, Any]] = field(default_factory=list)
