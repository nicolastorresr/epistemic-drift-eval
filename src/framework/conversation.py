"""
conversation.py
Multi-turn conversation manager for epistemic drift evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ResponseCategory(str, Enum):
    """Classification of a model response transition."""
    CORRECTIVE_REVISION = "corrective_revision"      # Wrong → correct
    PRODUCTIVE_UNCERTAINTY = "productive_uncertainty" # Acknowledges limits
    EPISTEMIC_DRIFT = "epistemic_drift"              # Correct → uncertain/wrong
    COMPLIANCE_REVERSAL = "compliance_reversal"       # Correct → explicitly wrong
    STABLE = "stable"                                 # Unchanged correct
    UNKNOWN = "unknown"


@dataclass
class Turn:
    """Represents a single turn in a conversation."""
    turn_id: int
    role: str                        # "user" | "assistant"
    content: str
    pressure_category: Optional[str] = None   # Which pressure type was applied
    pressure_intensity: Optional[str] = None
    factually_correct: Optional[bool] = None
    confidence: Optional[float] = None        # 0–1 self-reported confidence
    response_category: Optional[ResponseCategory] = None

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "pressure_category": self.pressure_category,
            "pressure_intensity": self.pressure_intensity,
            "factually_correct": self.factually_correct,
            "confidence": self.confidence,
            "response_category": self.response_category.value if self.response_category else None,
        }


@dataclass
class ConversationTrajectory:
    """
    Full conversation trajectory for a single query.
    Tracks the model's epistemic path under pressure.
    """
    trajectory_id: str
    query_id: str
    domain: str
    question: str
    reference_answer: str
    model_name: str
    turns: list[Turn] = field(default_factory=list)
    initial_response: Optional[str] = None
    initial_correct: Optional[bool] = None
    initial_confidence: Optional[float] = None
    final_response: Optional[str] = None
    final_correct: Optional[bool] = None
    reversal_turn: Optional[int] = None          # Turn at which reversal occurred
    pressure_sequence: list[str] = field(default_factory=list)

    @property
    def had_reversal(self) -> bool:
        return self.reversal_turn is not None

    @property
    def num_pressure_turns(self) -> int:
        return len([t for t in self.turns if t.role == "user" and t.pressure_category])

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def get_assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    def get_messages_for_api(self) -> list[dict]:
        """Return conversation as OpenAI/Anthropic-compatible messages list."""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "query_id": self.query_id,
            "domain": self.domain,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "model_name": self.model_name,
            "initial_response": self.initial_response,
            "initial_correct": self.initial_correct,
            "initial_confidence": self.initial_confidence,
            "final_response": self.final_response,
            "final_correct": self.final_correct,
            "reversal_turn": self.reversal_turn,
            "had_reversal": self.had_reversal,
            "num_pressure_turns": self.num_pressure_turns,
            "pressure_sequence": self.pressure_sequence,
            "turns": [t.to_dict() for t in self.turns],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
