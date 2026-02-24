"""Core evaluation engine for LLM benchmarking.

Provides the ModelEvaluator class that orchestrates benchmark execution
across multiple models, with support for both live API calls and simulated
evaluation for testing without API keys.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from src.benchmarks import BenchmarkCase, BenchmarkResult, BenchmarkSuite
from src.metrics import ModelComparisonReport, ModelMetrics, compare_models, compute_model_metrics


class ModelConfig(BaseModel):
    """Configuration for a single model.

    Attributes:
        name: Human-readable model name.
        provider: API provider (google, openai, anthropic).
        model_id: Provider-specific model identifier.
        cost_per_1k_input: Cost per 1000 input tokens in USD.
        cost_per_1k_output: Cost per 1000 output tokens in USD.
        max_tokens: Maximum tokens for model responses.
        description: Optional description of the model.
    """

    name: str
    provider: str
    model_id: str
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_tokens: int = 4096
    description: str = ""


class EvaluationConfig(BaseModel):
    """Configuration for evaluation runs.

    Attributes:
        models: List of model configurations to evaluate.
        default_num_runs: Number of times to run each benchmark.
        timeout_seconds: Timeout per model call in seconds.
        retry_attempts: Number of retries on failure.
        parallel_execution: Whether to run models in parallel.
    """

    models: list[ModelConfig] = Field(default_factory=list)
    default_num_runs: int = 3
    timeout_seconds: int = 30
    retry_attempts: int = 2
    parallel_execution: bool = False


@dataclass
class EvaluationResult:
    """Result from evaluating a single model across all benchmarks.

    Attributes:
        model_name: Name of the evaluated model.
        model_id: Provider-specific model identifier.
        provider: API provider name.
        scores: Dictionary of category to average score.
        overall_score: Overall accuracy score.
        latency_ms: Average latency in milliseconds.
        cost_per_1k_tokens: Average cost per 1000 tokens.
        benchmark_results: List of individual benchmark results.
        total_tokens: Total tokens generated.
        total_time_ms: Total evaluation time in milliseconds.
        metadata: Additional metadata about the evaluation.
    """

    model_name: str
    model_id: str
    provider: str
    scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    latency_ms: float = 0.0
    cost_per_1k_tokens: float = 0.0
    benchmark_results: list[BenchmarkResult] = field(default_factory=list)
    total_tokens: int = 0
    total_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelEvaluator:
    """Core evaluation engine that runs benchmarks across multiple models.

    Supports both live API evaluation and simulated/mock evaluation mode
    for testing without API keys.

    Args:
        config: EvaluationConfig with model and run settings.
        benchmark_suite: Optional BenchmarkSuite to use (defaults to built-in).
        simulate: If True, use simulated responses instead of real API calls.
    """

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        benchmark_suite: BenchmarkSuite | None = None,
        simulate: bool = True,
    ) -> None:
        self.config = config or EvaluationConfig()
        self.suite = benchmark_suite or BenchmarkSuite()
        self.simulate = simulate
        self._results: list[EvaluationResult] = []

    @classmethod
    def from_yaml(cls, config_path: str | Path, simulate: bool = True) -> ModelEvaluator:
        """Create an evaluator from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.
            simulate: Whether to run in simulation mode.

        Returns:
            Configured ModelEvaluator instance.
        """
        path = Path(config_path)
        with path.open() as f:
            raw = yaml.safe_load(f)

        models = [ModelConfig(**m) for m in raw.get("models", [])]
        eval_settings = raw.get("evaluation", {})

        config = EvaluationConfig(
            models=models,
            default_num_runs=eval_settings.get("default_num_runs", 3),
            timeout_seconds=eval_settings.get("timeout_seconds", 30),
            retry_attempts=eval_settings.get("retry_attempts", 2),
            parallel_execution=eval_settings.get("parallel_execution", False),
        )

        return cls(config=config, simulate=simulate)

    def evaluate_model(self, model_config: ModelConfig) -> EvaluationResult:
        """Evaluate a single model against all benchmarks in the suite.

        Args:
            model_config: Configuration of the model to evaluate.

        Returns:
            EvaluationResult with all benchmark scores and metrics.
        """
        benchmark_results: list[BenchmarkResult] = []
        category_scores: dict[str, list[float]] = {}

        for case in self.suite.cases:
            if self.simulate:
                output, latency_ms, token_count = self._simulate_response(
                    model_config, case
                )
            else:
                output, latency_ms, token_count = self._call_api(model_config, case)

            result = self.suite.evaluate_output(case, output, latency_ms, token_count)
            benchmark_results.append(result)

            cat_name = case.category.value
            if cat_name not in category_scores:
                category_scores[cat_name] = []
            category_scores[cat_name].append(result.score)

        # Calculate aggregate scores
        scores = {cat: sum(s) / len(s) for cat, s in category_scores.items()}
        all_scores = [r.score for r in benchmark_results]
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        latencies = [r.latency_ms for r in benchmark_results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        total_tokens = sum(r.token_count for r in benchmark_results)
        total_time = sum(r.latency_ms for r in benchmark_results)
        avg_cost = (model_config.cost_per_1k_input + model_config.cost_per_1k_output) / 2

        result = EvaluationResult(
            model_name=model_config.name,
            model_id=model_config.model_id,
            provider=model_config.provider,
            scores=scores,
            overall_score=overall,
            latency_ms=avg_latency,
            cost_per_1k_tokens=avg_cost,
            benchmark_results=benchmark_results,
            total_tokens=total_tokens,
            total_time_ms=total_time,
            metadata={
                "num_benchmarks": len(benchmark_results),
                "simulation_mode": self.simulate,
            },
        )

        self._results.append(result)
        return result

    def evaluate_all(self) -> list[EvaluationResult]:
        """Evaluate all configured models against the benchmark suite.

        Returns:
            List of EvaluationResult for each model.
        """
        results = []
        for model_config in self.config.models:
            result = self.evaluate_model(model_config)
            results.append(result)
        return results

    def get_comparison_report(self) -> ModelComparisonReport:
        """Generate a comparison report from all evaluated models.

        Returns:
            ModelComparisonReport with rankings across all metrics.

        Raises:
            ValueError: If no models have been evaluated yet.
        """
        if not self._results:
            raise ValueError("No evaluation results available. Run evaluate_all() first.")

        metrics_list: list[ModelMetrics] = []
        for result in self._results:
            benchmark_scores = [r.score for r in result.benchmark_results]
            benchmark_passed = [r.passed for r in result.benchmark_results]
            latencies = [r.latency_ms for r in result.benchmark_results]
            token_counts = [r.token_count for r in result.benchmark_results]
            expected_kw = [r.expected_keywords for r in result.benchmark_results]
            found_kw = [r.found_keywords for r in result.benchmark_results]
            outputs = [r.model_output for r in result.benchmark_results]

            # Get the model config for cost info
            model_cfg = next(
                (m for m in self.config.models if m.name == result.model_name),
                None,
            )
            cost_input = model_cfg.cost_per_1k_input if model_cfg else 0.0
            cost_output = model_cfg.cost_per_1k_output if model_cfg else 0.0

            category_results: dict[str, list[float]] = {}
            for br in result.benchmark_results:
                cat = br.category.value
                if cat not in category_results:
                    category_results[cat] = []
                category_results[cat].append(br.score)

            metrics = compute_model_metrics(
                model_name=result.model_name,
                benchmark_scores=benchmark_scores,
                benchmark_passed=benchmark_passed,
                latencies_ms=latencies,
                token_counts=token_counts,
                expected_keywords_list=expected_kw,
                found_keywords_list=found_kw,
                outputs=outputs,
                cost_per_1k_input=cost_input,
                cost_per_1k_output=cost_output,
                category_results=category_results,
            )
            metrics_list.append(metrics)

        return compare_models(metrics_list)

    @property
    def results(self) -> list[EvaluationResult]:
        """Access the stored evaluation results."""
        return self._results

    def clear_results(self) -> None:
        """Clear all stored evaluation results."""
        self._results.clear()

    def _simulate_response(
        self,
        model_config: ModelConfig,
        benchmark: BenchmarkCase,
    ) -> tuple[str, float, int]:
        """Generate a simulated response for testing without API keys.

        Uses deterministic seeding based on the model and benchmark to produce
        consistent but varied results across models.

        Args:
            model_config: The model configuration.
            benchmark: The benchmark case.

        Returns:
            Tuple of (output_text, latency_ms, token_count).
        """
        # Create deterministic seed from model + benchmark combination
        seed_str = f"{model_config.model_id}:{benchmark.name}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Simulate model quality based on cost (higher cost = generally better)
        cost_factor = model_config.cost_per_1k_input + model_config.cost_per_1k_output
        quality_base = min(0.95, 0.5 + cost_factor * 30)

        # Determine which keywords to include (simulating model accuracy)
        included_keywords = []
        for kw in benchmark.expected_output_contains:
            if rng.random() < quality_base:
                included_keywords.append(kw)

        # Build simulated output containing selected keywords
        output_parts = [f"Based on the input, here is my analysis regarding {benchmark.name}."]
        for kw in included_keywords:
            output_parts.append(f"The response includes information about {kw}.")
        output_parts.append("This concludes the evaluation response.")
        output = " ".join(output_parts)

        # Simulate latency: faster models (cheaper) tend to be faster
        base_latency = 200 + (cost_factor * 5000)
        latency_ms = base_latency + rng.gauss(0, base_latency * 0.15)
        latency_ms = max(50, latency_ms)

        # Simulate token count
        token_count = len(output.split()) + rng.randint(10, 50)

        return output, latency_ms, token_count

    def _call_api(
        self,
        model_config: ModelConfig,
        benchmark: BenchmarkCase,
    ) -> tuple[str, float, int]:
        """Call an actual model API. Currently delegates to simulation.

        In a production implementation, this would dispatch to the appropriate
        provider API (Google, OpenAI, Anthropic).

        Args:
            model_config: The model configuration.
            benchmark: The benchmark case.

        Returns:
            Tuple of (output_text, latency_ms, token_count).
        """
        # In production, this would route to the appropriate API client.
        # For now, fall back to simulation.
        return self._simulate_response(model_config, benchmark)


def load_config(config_path: str | Path) -> EvaluationConfig:
    """Load evaluation configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        EvaluationConfig with all model and evaluation settings.
    """
    path = Path(config_path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    models = [ModelConfig(**m) for m in raw.get("models", [])]
    eval_settings = raw.get("evaluation", {})

    return EvaluationConfig(
        models=models,
        default_num_runs=eval_settings.get("default_num_runs", 3),
        timeout_seconds=eval_settings.get("timeout_seconds", 30),
        retry_attempts=eval_settings.get("retry_attempts", 2),
        parallel_execution=eval_settings.get("parallel_execution", False),
    )
