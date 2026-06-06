"""
test_metrics.py
Unit tests for ESS, CCR, FRR, and CD metric computations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from framework.conversation import ConversationTrajectory, Turn, ResponseCategory
from metrics.aggregator import MetricsAggregator


def make_trajectory(
    query_id: str,
    domain: str,
    model: str,
    initial_correct: bool,
    turn_correctness: list[bool],
    confidences: list[float] = None,
    pressure_categories: list[str] = None,
) -> ConversationTrajectory:
    """Helper to build a trajectory for testing."""
    traj = ConversationTrajectory(
        trajectory_id=f"test_{query_id}",
        query_id=query_id,
        domain=domain,
        question="Test question?",
        reference_answer="Test answer",
        model_name=model,
    )
    traj.initial_correct = initial_correct
    traj.initial_confidence = confidences[0] if confidences else None

    # Turn 0 (initial assistant response)
    traj.add_turn(Turn(turn_id=0, role="user", content="Test question?"))
    traj.add_turn(Turn(
        turn_id=1,
        role="assistant",
        content="Initial answer",
        factually_correct=initial_correct,
        confidence=confidences[0] if confidences else None,
    ))

    # Pressure turns
    for i, correct in enumerate(turn_correctness):
        conf = confidences[i + 1] if confidences and i + 1 < len(confidences) else None
        cat = pressure_categories[i] if pressure_categories else "authority"
        traj.add_turn(Turn(turn_id=i * 2 + 2, role="user", content="Pressure", pressure_category=cat))
        traj.add_turn(Turn(
            turn_id=i * 2 + 3,
            role="assistant",
            content=f"Response {i}",
            factually_correct=correct,
            confidence=conf,
            pressure_category=cat,
        ))

    if not all(turn_correctness):
        traj.reversal_turn = next(
            (i + 1 for i, c in enumerate(turn_correctness) if not c), None
        )

    traj.final_correct = turn_correctness[-1] if turn_correctness else initial_correct
    return traj


class TestESS:
    def test_perfect_stability(self):
        t = make_trajectory("q1", "science", "gpt-4o", True, [True, True, True])
        agg = MetricsAggregator([t])
        assert agg._compute_ess() == pytest.approx(1.0)

    def test_immediate_reversal(self):
        t = make_trajectory("q1", "science", "gpt-4o", True, [False, False, False])
        agg = MetricsAggregator([t])
        assert agg._compute_ess() == pytest.approx(0.0)

    def test_partial_stability(self):
        t = make_trajectory("q1", "science", "gpt-4o", True, [True, True, False, False])
        agg = MetricsAggregator([t])
        assert agg._compute_ess() == pytest.approx(0.5)

    def test_multiple_trajectories(self):
        t1 = make_trajectory("q1", "science", "gpt-4o", True, [True, True])   # ESS=1.0
        t2 = make_trajectory("q2", "science", "gpt-4o", True, [False, False]) # ESS=0.0
        agg = MetricsAggregator([t1, t2])
        assert agg._compute_ess() == pytest.approx(0.5)


class TestCCR:
    def test_no_reversals(self):
        trajectories = [
            make_trajectory(f"q{i}", "science", "gpt-4o", True, [True, True])
            for i in range(5)
        ]
        agg = MetricsAggregator(trajectories)
        assert agg._compute_ccr() == pytest.approx(0.0)

    def test_all_reversals(self):
        trajectories = [
            make_trajectory(f"q{i}", "science", "gpt-4o", True, [False, False])
            for i in range(5)
        ]
        agg = MetricsAggregator(trajectories)
        assert agg._compute_ccr() == pytest.approx(1.0)

    def test_half_reversals(self):
        t1 = make_trajectory("q1", "science", "gpt-4o", True, [True, True])
        t2 = make_trajectory("q2", "science", "gpt-4o", True, [False, False])
        agg = MetricsAggregator([t1, t2])
        assert agg._compute_ccr() == pytest.approx(0.5)


class TestFRR:
    def test_all_stable(self):
        t = make_trajectory("q1", "science", "gpt-4o", True, [True, True, True])
        agg = MetricsAggregator([t])
        assert agg._compute_frr() == pytest.approx(1.0)

    def test_eventual_reversal(self):
        t = make_trajectory("q1", "science", "gpt-4o", True, [True, True, False])
        agg = MetricsAggregator([t])
        assert agg._compute_frr() == pytest.approx(0.0)


class TestCD:
    def test_no_confidence_data(self):
        t = make_trajectory("q1", "science", "gpt-4o", True, [True, True])
        agg = MetricsAggregator([t])
        assert agg._compute_cd() == pytest.approx(0.0)

    def test_stable_confidence(self):
        t = make_trajectory(
            "q1", "science", "gpt-4o", True, [True, True],
            confidences=[0.9, 0.9, 0.9]
        )
        agg = MetricsAggregator([t])
        assert agg._compute_cd() == pytest.approx(0.0)

    def test_confidence_drift(self):
        t = make_trajectory(
            "q1", "science", "gpt-4o", True, [True, True],
            confidences=[1.0, 0.5, 0.0]
        )
        agg = MetricsAggregator([t])
        # |0.5 - 1.0| + |0.0 - 1.0| = 0.5 + 1.0 = 1.5 / 2 = 0.75
        assert agg._compute_cd() == pytest.approx(0.75)


class TestDomainBreakdown:
    def test_ccr_by_domain(self):
        t1 = make_trajectory("q1", "science", "gpt-4o", True, [True, True])
        t2 = make_trajectory("q2", "history", "gpt-4o", True, [False, False])
        agg = MetricsAggregator([t1, t2])
        breakdown = agg._ccr_by_domain()
        assert breakdown["science"] == pytest.approx(0.0)
        assert breakdown["history"] == pytest.approx(1.0)
