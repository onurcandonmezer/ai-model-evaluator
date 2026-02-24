"""Tests for the benchmarks module."""

from __future__ import annotations

import pytest

from src.benchmarks import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkSuite,
    Difficulty,
)


class TestBenchmarkSuite:
    """Tests for the BenchmarkSuite class."""

    def test_default_suite_has_cases(self) -> None:
        """Default suite should contain at least 20 benchmark cases."""
        suite = BenchmarkSuite()
        assert suite.total_cases >= 20

    def test_default_suite_covers_all_categories(self) -> None:
        """Default suite should cover all benchmark categories."""
        suite = BenchmarkSuite()
        categories = suite.categories
        for cat in BenchmarkCategory:
            assert cat in categories, f"Category {cat.value} missing from default suite"

    def test_filter_by_category(self) -> None:
        """Filtering by category should return only matching cases."""
        suite = BenchmarkSuite()
        qa_cases = suite.get_by_category(BenchmarkCategory.QA)
        assert len(qa_cases) > 0
        for case in qa_cases:
            assert case.category == BenchmarkCategory.QA

    def test_filter_by_difficulty(self) -> None:
        """Filtering by difficulty should return only matching cases."""
        suite = BenchmarkSuite()
        easy_cases = suite.get_by_difficulty(Difficulty.EASY)
        assert len(easy_cases) > 0
        for case in easy_cases:
            assert case.difficulty == Difficulty.EASY

    def test_custom_suite(self) -> None:
        """Suite can be initialized with custom cases."""
        custom_cases = [
            BenchmarkCase(
                name="custom_test",
                category=BenchmarkCategory.QA,
                input_text="What is 2+2?",
                expected_output_contains=["4"],
                difficulty=Difficulty.EASY,
            ),
        ]
        suite = BenchmarkSuite(cases=custom_cases)
        assert suite.total_cases == 1
        assert suite.cases[0].name == "custom_test"

    def test_evaluate_output_full_match(self) -> None:
        """Evaluating output with all keywords present should score 1.0."""
        suite = BenchmarkSuite()
        case = BenchmarkCase(
            name="test_case",
            category=BenchmarkCategory.QA,
            input_text="Test input",
            expected_output_contains=["keyword1", "keyword2"],
        )
        output = "This output contains keyword1 and keyword2 in the response."
        result = suite.evaluate_output(case, output, latency_ms=100.0, token_count=15)

        assert result.score == 1.0
        assert result.passed is True
        assert len(result.found_keywords) == 2

    def test_evaluate_output_partial_match(self) -> None:
        """Evaluating output with some keywords should score proportionally."""
        suite = BenchmarkSuite()
        case = BenchmarkCase(
            name="test_case",
            category=BenchmarkCategory.QA,
            input_text="Test input",
            expected_output_contains=["alpha", "beta", "gamma", "delta"],
        )
        output = "This output mentions alpha and beta but not the others."
        result = suite.evaluate_output(case, output, latency_ms=200.0, token_count=12)

        assert result.score == 0.5
        assert result.passed is True
        assert len(result.found_keywords) == 2

    def test_evaluate_output_no_match(self) -> None:
        """Evaluating output with no keywords should score 0.0."""
        suite = BenchmarkSuite()
        case = BenchmarkCase(
            name="test_case",
            category=BenchmarkCategory.QA,
            input_text="Test input",
            expected_output_contains=["missing1", "missing2"],
        )
        output = "This output has nothing relevant."
        result = suite.evaluate_output(case, output, latency_ms=150.0, token_count=8)

        assert result.score == 0.0
        assert result.passed is False
        assert len(result.found_keywords) == 0

    def test_evaluate_output_case_insensitive(self) -> None:
        """Keyword matching should be case-insensitive."""
        suite = BenchmarkSuite()
        case = BenchmarkCase(
            name="test_case",
            category=BenchmarkCategory.QA,
            input_text="Test input",
            expected_output_contains=["Python", "JAVA"],
        )
        output = "This mentions python and java in lowercase."
        result = suite.evaluate_output(case, output, latency_ms=100.0, token_count=10)

        assert result.score == 1.0
        assert result.passed is True


class TestBenchmarkCase:
    """Tests for the BenchmarkCase dataclass."""

    def test_benchmark_case_creation(self) -> None:
        """BenchmarkCase should be created with required fields."""
        case = BenchmarkCase(
            name="test",
            category=BenchmarkCategory.SUMMARIZATION,
            input_text="Summarize this.",
            expected_output_contains=["summary"],
        )
        assert case.name == "test"
        assert case.category == BenchmarkCategory.SUMMARIZATION
        assert case.difficulty == Difficulty.MEDIUM  # default
        assert case.max_tokens == 512  # default

    def test_benchmark_case_is_frozen(self) -> None:
        """BenchmarkCase should be immutable (frozen dataclass)."""
        case = BenchmarkCase(
            name="test",
            category=BenchmarkCategory.QA,
            input_text="input",
            expected_output_contains=["answer"],
        )
        with pytest.raises(AttributeError):
            case.name = "modified"  # type: ignore[misc]


class TestBenchmarkResult:
    """Tests for the BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self) -> None:
        """BenchmarkResult should store all evaluation data."""
        result = BenchmarkResult(
            benchmark_name="test",
            category=BenchmarkCategory.QA,
            model_output="The answer is 42.",
            expected_keywords=["42"],
            found_keywords=["42"],
            latency_ms=150.5,
            token_count=6,
            passed=True,
            score=1.0,
        )
        assert result.benchmark_name == "test"
        assert result.passed is True
        assert result.score == 1.0
        assert result.latency_ms == 150.5
