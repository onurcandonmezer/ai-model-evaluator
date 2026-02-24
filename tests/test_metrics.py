"""Tests for the metrics module."""

from __future__ import annotations

import pytest

from src.metrics import (
    LatencyPercentiles,
    ModelComparisonReport,
    ModelMetrics,
    accuracy_score,
    compare_models,
    compute_model_metrics,
    cost_per_quality_score,
    f1_score,
    format_metrics_table,
    hallucination_rate,
    latency_percentiles,
    throughput_tokens_per_second,
)


class TestAccuracyScore:
    """Tests for accuracy_score function."""

    def test_perfect_accuracy(self) -> None:
        assert accuracy_score(10, 10) == 1.0

    def test_zero_accuracy(self) -> None:
        assert accuracy_score(0, 10) == 0.0

    def test_partial_accuracy(self) -> None:
        assert accuracy_score(7, 10) == pytest.approx(0.7)

    def test_empty_total(self) -> None:
        assert accuracy_score(0, 0) == 0.0


class TestF1Score:
    """Tests for f1_score function."""

    def test_perfect_f1(self) -> None:
        assert f1_score(10, 0, 0) == 1.0

    def test_zero_f1(self) -> None:
        assert f1_score(0, 5, 5) == 0.0

    def test_balanced_f1(self) -> None:
        # precision = 8/10 = 0.8, recall = 8/12 = 0.667
        # f1 = 2 * 0.8 * 0.667 / (0.8 + 0.667) = 0.727
        result = f1_score(8, 2, 4)
        assert 0.72 < result < 0.73

    def test_all_zeros(self) -> None:
        assert f1_score(0, 0, 0) == 0.0


class TestLatencyPercentiles:
    """Tests for latency_percentiles function."""

    def test_basic_percentiles(self) -> None:
        latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = latency_percentiles(latencies)
        assert isinstance(result, LatencyPercentiles)
        assert result.p50 == pytest.approx(300.0)
        assert result.min == 100.0
        assert result.max == 500.0
        assert result.mean == pytest.approx(300.0)

    def test_single_value(self) -> None:
        result = latency_percentiles([150.0])
        assert result.p50 == 150.0
        assert result.p95 == 150.0
        assert result.p99 == 150.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            latency_percentiles([])


class TestCostPerQualityScore:
    """Tests for cost_per_quality_score function."""

    def test_normal_ratio(self) -> None:
        result = cost_per_quality_score(0.01, 0.8)
        assert result == pytest.approx(0.0125)

    def test_zero_quality(self) -> None:
        result = cost_per_quality_score(0.01, 0.0)
        assert result == float("inf")

    def test_zero_cost(self) -> None:
        result = cost_per_quality_score(0.0, 0.8)
        assert result == 0.0


class TestHallucinationRate:
    """Tests for hallucination_rate function."""

    def test_no_hallucination(self) -> None:
        expected = [["keyword1", "keyword2"]]
        found = [["keyword1", "keyword2"]]
        outputs = ["A response with keyword1 and keyword2."]
        rate = hallucination_rate(expected, found, outputs)
        assert rate == 0.0

    def test_full_hallucination_from_missing_keywords(self) -> None:
        expected = [["a", "b", "c", "d", "e"]]
        found = [[]]  # none found
        outputs = ["A completely irrelevant output."]
        rate = hallucination_rate(expected, found, outputs)
        assert rate == 1.0

    def test_hallucination_from_indicators(self) -> None:
        expected = [["answer"]]
        found = [["answer"]]
        outputs = ["I don't know the answer but I'll try."]
        rate = hallucination_rate(expected, found, outputs)
        assert rate == 1.0

    def test_empty_outputs(self) -> None:
        rate = hallucination_rate([], [], [])
        assert rate == 0.0


class TestThroughput:
    """Tests for throughput_tokens_per_second function."""

    def test_normal_throughput(self) -> None:
        # 1000 tokens in 2000ms = 500 tokens/sec
        result = throughput_tokens_per_second(1000, 2000.0)
        assert result == pytest.approx(500.0)

    def test_zero_time(self) -> None:
        result = throughput_tokens_per_second(100, 0.0)
        assert result == 0.0


class TestComputeModelMetrics:
    """Tests for compute_model_metrics function."""

    def test_compute_metrics_basic(self) -> None:
        metrics = compute_model_metrics(
            model_name="test-model",
            benchmark_scores=[1.0, 0.5, 0.75],
            benchmark_passed=[True, True, True],
            latencies_ms=[100.0, 200.0, 150.0],
            token_counts=[50, 60, 55],
            expected_keywords_list=[["a"], ["b"], ["c"]],
            found_keywords_list=[["a"], ["b"], ["c"]],
            outputs=["output a", "output b", "output c"],
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
        )
        assert metrics.model_name == "test-model"
        assert metrics.accuracy_score == 1.0
        assert metrics.total_benchmarks == 3
        assert metrics.passed_benchmarks == 3
        assert metrics.latency.mean == pytest.approx(150.0)
        assert metrics.throughput_tokens_per_second > 0


class TestCompareModels:
    """Tests for compare_models function."""

    def _make_metrics(self, name: str, accuracy: float, latency_p50: float,
                      cost: float) -> ModelMetrics:
        return ModelMetrics(
            model_name=name,
            accuracy_score=accuracy,
            f1_score=accuracy,
            latency=LatencyPercentiles(
                p50=latency_p50, p95=latency_p50 * 1.5,
                p99=latency_p50 * 2.0, mean=latency_p50,
                min=latency_p50 * 0.5, max=latency_p50 * 2.5,
            ),
            cost_per_1k_tokens=cost,
            cost_per_quality_score=cost / accuracy if accuracy > 0 else float("inf"),
            hallucination_rate=0.1,
            throughput_tokens_per_second=500.0,
            total_benchmarks=10,
            passed_benchmarks=int(accuracy * 10),
        )

    def test_compare_produces_rankings(self) -> None:
        models = [
            self._make_metrics("model-a", 0.9, 200.0, 0.01),
            self._make_metrics("model-b", 0.7, 100.0, 0.005),
            self._make_metrics("model-c", 0.8, 150.0, 0.008),
        ]
        report = compare_models(models)
        assert isinstance(report, ModelComparisonReport)
        assert len(report.rankings) == 3
        assert report.rankings[0].rank == 1
        assert report.rankings[1].rank == 2
        assert report.rankings[2].rank == 3

    def test_compare_best_selections(self) -> None:
        models = [
            self._make_metrics("accurate", 0.95, 500.0, 0.02),
            self._make_metrics("fast", 0.7, 50.0, 0.01),
            self._make_metrics("cheap", 0.6, 300.0, 0.001),
        ]
        report = compare_models(models)
        assert report.best_accuracy == "accurate"
        assert report.best_latency == "fast"
        assert report.best_cost == "cheap"

    def test_compare_empty_models(self) -> None:
        report = compare_models([])
        assert report.best_overall == "N/A"
        assert len(report.rankings) == 0


class TestFormatMetricsTable:
    """Tests for format_metrics_table function."""

    def test_format_produces_dict(self) -> None:
        metrics = ModelMetrics(
            model_name="test",
            accuracy_score=0.85,
            f1_score=0.82,
            latency=LatencyPercentiles(
                p50=200.0, p95=350.0, p99=500.0,
                mean=220.0, min=100.0, max=600.0,
            ),
            cost_per_1k_tokens=0.005,
            cost_per_quality_score=0.00588,
            hallucination_rate=0.05,
            throughput_tokens_per_second=450.0,
            total_benchmarks=20,
            passed_benchmarks=17,
        )
        table = format_metrics_table(metrics)
        assert table["Model"] == "test"
        assert "85.0%" in table["Accuracy"]
        assert "17/20" in table["Passed"]
