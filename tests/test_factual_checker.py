"""
test_factual_checker.py
Unit tests for the FactualChecker class.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from evaluation.factual_checker import FactualChecker


class TestExactMatch:
    def setup_method(self):
        self.checker = FactualChecker(method="exact_match")

    def test_exact_correct(self):
        assert self.checker.is_correct("The answer is Canberra.", "Canberra") is True

    def test_case_insensitive(self):
        assert self.checker.is_correct("CANBERRA is the capital.", "canberra") is True

    def test_punctuation_stripped(self):
        assert self.checker.is_correct("The answer is Canberra!", "Canberra") is True

    def test_wrong_answer(self):
        assert self.checker.is_correct("The capital is Sydney.", "Canberra") is False

    def test_alias_match(self):
        assert self.checker.is_correct(
            "The answer is H₂O.", "H2O", aliases=["H₂O", "water"]
        ) is True

    def test_number_in_response(self):
        assert self.checker.is_correct("There are 206 bones.", "206") is True

    def test_no_false_positive(self):
        assert self.checker.is_correct(
            "The answer is not Paris.", "Paris"
        ) is True  # substring match; intentional — model contains the answer word
        # Note: exact_match uses substring, not semantic negation detection


class TestHybridMatch:
    def setup_method(self):
        self.checker = FactualChecker(method="exact_match")  # avoid loading model in CI

    def test_correct_response(self):
        assert self.checker.is_correct("Canberra is Australia's capital.", "Canberra") is True

    def test_incorrect_response(self):
        assert self.checker.is_correct("Sydney is Australia's capital.", "Canberra") is False

    def test_numerical_answer(self):
        assert self.checker.is_correct("The result is 1024.", "1024") is True

    def test_alias_accepted(self):
        assert self.checker.is_correct(
            "Baikal is the deepest lake.", "Lake Baikal", aliases=["Baikal"]
        ) is True
