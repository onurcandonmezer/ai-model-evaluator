"""Metric calculations for LLM evaluation.

Provides functions to compute accuracy, F1, latency percentiles, cost analysis,
hallucination rates, throughput, and comprehensive model comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LatencyPercentiles:
    """Latency statistics at various percentiles.

    Attributes:
        p50: Median latency in milliseconds.
        p95: 95th percentile latency in milliseconds.
        p99: 99th percentile latency in milliseconds.
        mean: Mean latency in milliseconds.
        min: Minimum latency in milliseconds.
        max: Maximum latency in milliseconds.
    """

    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float


@dataclass
class ModelMetrics:
    """Comprehensive metrics for a single model.

    Attributes:
        model_name: Name of the evaluated model.
        accuracy_score: Overall accuracy from 0.0 to 1.0.
        f1_score: F1 score (harmonic mean of precision and recall).
        latency: Latency statistics at various percentiles.
        cost_per_1k_tokens: Cost per 1000 tokens processed.
        cost_per_quality_score: Cost divided by quality score ratio.
        hallucination_rate: Rate of hallucinated content from 0.0 to 1.0.
        throughput_tokens_per_second: Token generation throughput.
        category_scores: Accuracy breakdown by benchmark category.
        total_benchmarks: Total number of benchmarks evaluated.
        passed_benchmarks: Number of benchmarks passed.
    """

    model_name: str
    accuracy_score: float
    f1_score: float
    latency: LatencyPercentiles
    cost_per_1k_tokens: float
    cost_per_quality_score: float
    hallucination_rate: float
    throughput_tokens_per_second: float
    category_scores: dict[str, float] = field(default_factory=dict)
    total_benchmarks: int = 0
    passed_benchmarks: int = 0


@dataclass
class ModelRanking:
    """Ranking entry for a model in comparison.

    Attributes:
        model_name: Name of the model.
        overall_score: Weighted overall score from 0.0 to 1.0.
        rank: Numerical rank position (1 = best).
        scores: Individual metric scores used in ranking.
    """

    model_name: str
    overall_score: float
    rank: int
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelComparisonReport:
    """Complete comparison report across multiple models.

    Attributes:
        models: List of ModelMetrics for each evaluated model.
        rankings: Ordered list of ModelRanking entries.
        best_accuracy: Name of model with best accuracy.
        best_latency: Name of model with best latency.
        best_cost: Name of model with best cost efficiency.
        best_overall: Name of model with best overall score.
    """

    models: list[ModelMetrics]
    rankings: list[ModelRanking]
    best_accuracy: str
    best_latency: str
    best_cost: str
    best_overall: str


def accuracy_score(passed: int, total: int) -> float:
    """Calculate accuracy as the ratio of passed benchmarks to total.

    Args:
        passed: Number of benchmarks passed.
        total: Total number of benchmarks.

    Returns:
        Accuracy score from 0.0 to 1.0.
    """
    if total == 0:
        return 0.0
    return passed / total


def f1_score(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> float:
    """Calculate F1 score from precision and recall components.

    Args:
        true_positives: Number of correctly identified positives.
        false_positives: Number of incorrectly identified positives.
        false_negatives: Number of missed positives.

    Returns:
        F1 score from 0.0 to 1.0.
    """
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def latency_percentiles(latencies_ms: list[float]) -> LatencyPercentiles:
    """Calculate latency percentiles from a list of latency measurements.

    Args:
        latencies_ms: List of latency values in milliseconds.

    Returns:
        LatencyPercentiles with p50, p95, p99, mean, min, max.

    Raises:
        ValueError: If latencies list is empty.
    """
    if not latencies_ms:
        raise ValueError("Cannot calculate percentiles from empty latency list.")

    arr = np.array(latencies_ms)
    return LatencyPercentiles(
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        mean=float(np.mean(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
    )


def cost_per_quality_score(
    cost_per_1k_tokens: float,
    quality_score: float,
) -> float:
    """Calculate the cost-to-quality ratio.

    Lower values indicate better cost efficiency relative to quality.

    Args:
        cost_per_1k_tokens: Cost per 1000 tokens.
        quality_score: Quality score from 0.0 to 1.0.

    Returns:
        Cost per quality point. Returns infinity if quality is zero.
    """
    if quality_score <= 0:
        return float("inf")
    return cost_per_1k_tokens / quality_score


def hallucination_rate(
    expected_keywords_list: list[list[str]],
    found_keywords_list: list[list[str]],
    outputs: list[str],
    hallucination_indicators: list[str] | None = None,
) -> float:
    """Calculate hallucination rate based on keyword analysis.

    Hallucinations are detected when the model produces content with
    unexpected keywords or misses expected keywords significantly.

    Args:
        expected_keywords_list: List of expected keyword lists per benchmark.
        found_keywords_list: List of found keyword lists per benchmark.
        outputs: List of model output strings.
        hallucination_indicators: Optional list of phrases that indicate hallucination.

    Returns:
        Hallucination rate from 0.0 to 1.0.
    """
    if not outputs:
        return 0.0

    if hallucination_indicators is None:
        hallucination_indicators = [
            "i don't know",
            "i cannot",
            "as an ai",
            "i'm not sure",
            "i apologize",
            "make up",
            "fabricat",
        ]

    hallucination_count = 0
    total_checks = len(outputs)

    for i, output in enumerate(outputs):
        output_lower = output.lower()

        # Check for excessive missed keywords (over-hallucination)
        if i < len(expected_keywords_list) and i < len(found_keywords_list):
            expected = expected_keywords_list[i]
            found = found_keywords_list[i]
            if len(expected) > 0:
                miss_rate = 1.0 - (len(found) / len(expected))
                if miss_rate > 0.7:
                    hallucination_count += 1
                    continue

        # Check for hallucination indicator phrases
        for indicator in hallucination_indicators:
            if indicator.lower() in output_lower:
                hallucination_count += 1
                break

    return hallucination_count / total_checks


def throughput_tokens_per_second(
    total_tokens: int,
    total_time_ms: float,
) -> float:
    """Calculate token generation throughput.

    Args:
        total_tokens: Total number of tokens generated.
        total_time_ms: Total time in milliseconds.

    Returns:
        Tokens per second throughput.
    """
    if total_time_ms <= 0:
        return 0.0
    return (total_tokens / total_time_ms) * 1000


def compute_model_metrics(
    model_name: str,
    benchmark_scores: list[float],
    benchmark_passed: list[bool],
    latencies_ms: list[float],
    token_counts: list[int],
    expected_keywords_list: list[list[str]],
    found_keywords_list: list[list[str]],
    outputs: list[str],
    cost_per_1k_input: float,
    cost_per_1k_output: float,
    category_results: dict[str, list[float]] | None = None,
) -> ModelMetrics:
    """Compute comprehensive metrics for a model from benchmark results.

    Args:
        model_name: Name of the model.
        benchmark_scores: List of scores per benchmark (0.0 to 1.0).
        benchmark_passed: List of pass/fail booleans per benchmark.
        latencies_ms: List of latency measurements in milliseconds.
        token_counts: List of token counts per response.
        expected_keywords_list: Expected keywords per benchmark.
        found_keywords_list: Found keywords per benchmark.
        outputs: Raw model outputs.
        cost_per_1k_input: Cost per 1000 input tokens.
        cost_per_1k_output: Cost per 1000 output tokens.
        category_results: Optional dict mapping category names to score lists.

    Returns:
        ModelMetrics with all computed values.
    """
    total = len(benchmark_scores)
    passed = sum(benchmark_passed)

    acc = accuracy_score(passed, total)

    # For F1: treat each benchmark as a classification decision
    tp = passed
    fp = 0  # In benchmark context, false positives are not directly applicable
    fn = total - passed
    f1 = f1_score(tp, fp, fn)

    latency = (
        latency_percentiles(latencies_ms)
        if latencies_ms
        else LatencyPercentiles(p50=0, p95=0, p99=0, mean=0, min=0, max=0)
    )

    avg_cost = (cost_per_1k_input + cost_per_1k_output) / 2
    cpq = cost_per_quality_score(avg_cost, acc)

    h_rate = hallucination_rate(expected_keywords_list, found_keywords_list, outputs)

    total_tokens = sum(token_counts)
    total_time = sum(latencies_ms)
    throughput = throughput_tokens_per_second(total_tokens, total_time)

    cat_scores: dict[str, float] = {}
    if category_results:
        for cat, scores in category_results.items():
            cat_scores[cat] = float(np.mean(scores)) if scores else 0.0

    return ModelMetrics(
        model_name=model_name,
        accuracy_score=acc,
        f1_score=f1,
        latency=latency,
        cost_per_1k_tokens=avg_cost,
        cost_per_quality_score=cpq,
        hallucination_rate=h_rate,
        throughput_tokens_per_second=throughput,
        category_scores=cat_scores,
        total_benchmarks=total,
        passed_benchmarks=passed,
    )


def compare_models(
    model_metrics_list: list[ModelMetrics],
    weights: dict[str, float] | None = None,
) -> ModelComparisonReport:
    """Compare multiple models and produce a ranked comparison report.

    Args:
        model_metrics_list: List of ModelMetrics for each model.
        weights: Optional dict of metric weights for overall scoring.
                 Keys: accuracy, latency, cost, hallucination, throughput.

    Returns:
        ModelComparisonReport with rankings and best-in-class selections.
    """
    if weights is None:
        weights = {
            "accuracy": 0.35,
            "latency": 0.20,
            "cost": 0.20,
            "hallucination": 0.15,
            "throughput": 0.10,
        }

    if not model_metrics_list:
        return ModelComparisonReport(
            models=[],
            rankings=[],
            best_accuracy="N/A",
            best_latency="N/A",
            best_cost="N/A",
            best_overall="N/A",
        )

    # Normalize metrics for ranking (0.0 to 1.0 scale, higher is better)
    max_accuracy = max(m.accuracy_score for m in model_metrics_list) or 1.0
    max_latency = max(m.latency.p50 for m in model_metrics_list) or 1.0
    max_cost = max(m.cost_per_1k_tokens for m in model_metrics_list) or 1.0
    max_throughput = max(m.throughput_tokens_per_second for m in model_metrics_list) or 1.0

    rankings: list[ModelRanking] = []
    for model in model_metrics_list:
        scores: dict[str, float] = {
            "accuracy": model.accuracy_score / max_accuracy if max_accuracy > 0 else 0,
            "latency": 1.0 - (model.latency.p50 / max_latency) if max_latency > 0 else 0,
            "cost": 1.0 - (model.cost_per_1k_tokens / max_cost) if max_cost > 0 else 0,
            "hallucination": 1.0 - model.hallucination_rate,
            "throughput": (
                model.throughput_tokens_per_second / max_throughput if max_throughput > 0 else 0
            ),
        }

        overall = sum(scores.get(k, 0) * v for k, v in weights.items())
        rankings.append(
            ModelRanking(
                model_name=model.model_name,
                overall_score=overall,
                rank=0,
                scores=scores,
            )
        )

    # Sort by overall score descending and assign ranks
    rankings.sort(key=lambda r: r.overall_score, reverse=True)
    for i, ranking in enumerate(rankings):
        ranking.rank = i + 1

    # Find best-in-class
    best_accuracy = max(model_metrics_list, key=lambda m: m.accuracy_score).model_name
    best_latency = min(model_metrics_list, key=lambda m: m.latency.p50).model_name
    best_cost = min(model_metrics_list, key=lambda m: m.cost_per_1k_tokens).model_name
    best_overall = rankings[0].model_name

    return ModelComparisonReport(
        models=model_metrics_list,
        rankings=rankings,
        best_accuracy=best_accuracy,
        best_latency=best_latency,
        best_cost=best_cost,
        best_overall=best_overall,
    )


def format_metrics_table(metrics: ModelMetrics) -> dict[str, Any]:
    """Convert ModelMetrics to a flat dictionary for table display.

    Args:
        metrics: ModelMetrics instance.

    Returns:
        Dictionary with human-readable metric keys and values.
    """
    return {
        "Model": metrics.model_name,
        "Accuracy": f"{metrics.accuracy_score:.1%}",
        "F1 Score": f"{metrics.f1_score:.3f}",
        "Latency P50 (ms)": f"{metrics.latency.p50:.1f}",
        "Latency P95 (ms)": f"{metrics.latency.p95:.1f}",
        "Cost/1K Tokens": f"${metrics.cost_per_1k_tokens:.4f}",
        "Hallucination Rate": f"{metrics.hallucination_rate:.1%}",
        "Throughput (tok/s)": f"{metrics.throughput_tokens_per_second:.1f}",
        "Passed": f"{metrics.passed_benchmarks}/{metrics.total_benchmarks}",
    }
