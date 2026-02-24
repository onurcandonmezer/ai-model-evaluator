"""Tests for the evaluator module."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmarks import BenchmarkCase, BenchmarkCategory, BenchmarkSuite, Difficulty
from src.evaluator import (
    EvaluationConfig,
    EvaluationResult,
    ModelConfig,
    ModelEvaluator,
    load_config,
)


@pytest.fixture
def sample_model_config() -> ModelConfig:
    """Create a sample model configuration."""
    return ModelConfig(
        name="Test Model",
        provider="test",
        model_id="test-model-v1",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        max_tokens=1024,
        description="A test model for unit testing.",
    )


@pytest.fixture
def sample_evaluation_config(sample_model_config: ModelConfig) -> EvaluationConfig:
    """Create a sample evaluation configuration."""
    return EvaluationConfig(
        models=[
            sample_model_config,
            ModelConfig(
                name="Test Model 2",
                provider="test",
                model_id="test-model-v2",
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.01,
            ),
        ],
        default_num_runs=1,
        timeout_seconds=10,
    )


@pytest.fixture
def small_suite() -> BenchmarkSuite:
    """Create a small benchmark suite for fast testing."""
    return BenchmarkSuite(
        cases=[
            BenchmarkCase(
                name="simple_qa",
                category=BenchmarkCategory.QA,
                input_text="What is 2+2?",
                expected_output_contains=["4", "four"],
                difficulty=Difficulty.EASY,
            ),
            BenchmarkCase(
                name="simple_classification",
                category=BenchmarkCategory.CLASSIFICATION,
                input_text="Is this positive? 'Great product!'",
                expected_output_contains=["positive"],
                difficulty=Difficulty.EASY,
            ),
            BenchmarkCase(
                name="simple_code",
                category=BenchmarkCategory.CODE_GENERATION,
                input_text="Write a hello world function in Python.",
                expected_output_contains=["def", "print", "hello"],
                difficulty=Difficulty.EASY,
            ),
        ]
    )


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_model_config_creation(self) -> None:
        config = ModelConfig(
            name="GPT-4o",
            provider="openai",
            model_id="gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
        )
        assert config.name == "GPT-4o"
        assert config.provider == "openai"
        assert config.max_tokens == 4096  # default

    def test_model_config_defaults(self) -> None:
        config = ModelConfig(
            name="Test",
            provider="test",
            model_id="test-v1",
        )
        assert config.cost_per_1k_input == 0.0
        assert config.cost_per_1k_output == 0.0
        assert config.description == ""


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""

    def test_default_config(self) -> None:
        config = EvaluationConfig()
        assert config.models == []
        assert config.default_num_runs == 3
        assert config.timeout_seconds == 30


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def test_evaluator_creation(self, sample_evaluation_config: EvaluationConfig) -> None:
        evaluator = ModelEvaluator(config=sample_evaluation_config, simulate=True)
        assert evaluator.simulate is True
        assert len(evaluator.config.models) == 2

    def test_evaluate_single_model(
        self,
        sample_model_config: ModelConfig,
        small_suite: BenchmarkSuite,
    ) -> None:
        config = EvaluationConfig(models=[sample_model_config])
        evaluator = ModelEvaluator(config=config, benchmark_suite=small_suite, simulate=True)
        result = evaluator.evaluate_model(sample_model_config)

        assert isinstance(result, EvaluationResult)
        assert result.model_name == "Test Model"
        assert result.provider == "test"
        assert len(result.benchmark_results) == 3
        assert result.total_tokens > 0
        assert result.total_time_ms > 0

    def test_evaluate_all_models(
        self,
        sample_evaluation_config: EvaluationConfig,
        small_suite: BenchmarkSuite,
    ) -> None:
        evaluator = ModelEvaluator(
            config=sample_evaluation_config,
            benchmark_suite=small_suite,
            simulate=True,
        )
        results = evaluator.evaluate_all()

        assert len(results) == 2
        assert results[0].model_name == "Test Model"
        assert results[1].model_name == "Test Model 2"

    def test_comparison_report(
        self,
        sample_evaluation_config: EvaluationConfig,
        small_suite: BenchmarkSuite,
    ) -> None:
        evaluator = ModelEvaluator(
            config=sample_evaluation_config,
            benchmark_suite=small_suite,
            simulate=True,
        )
        evaluator.evaluate_all()
        report = evaluator.get_comparison_report()

        assert len(report.rankings) == 2
        assert report.rankings[0].rank == 1
        assert report.rankings[1].rank == 2
        assert report.best_overall != ""

    def test_comparison_report_before_evaluation_raises(self) -> None:
        evaluator = ModelEvaluator(simulate=True)
        with pytest.raises(ValueError, match="No evaluation results"):
            evaluator.get_comparison_report()

    def test_clear_results(
        self,
        sample_model_config: ModelConfig,
        small_suite: BenchmarkSuite,
    ) -> None:
        config = EvaluationConfig(models=[sample_model_config])
        evaluator = ModelEvaluator(config=config, benchmark_suite=small_suite, simulate=True)
        evaluator.evaluate_model(sample_model_config)
        assert len(evaluator.results) == 1

        evaluator.clear_results()
        assert len(evaluator.results) == 0

    def test_simulated_responses_are_deterministic(
        self,
        sample_model_config: ModelConfig,
        small_suite: BenchmarkSuite,
    ) -> None:
        config = EvaluationConfig(models=[sample_model_config])

        evaluator1 = ModelEvaluator(config=config, benchmark_suite=small_suite, simulate=True)
        result1 = evaluator1.evaluate_model(sample_model_config)

        evaluator2 = ModelEvaluator(config=config, benchmark_suite=small_suite, simulate=True)
        result2 = evaluator2.evaluate_model(sample_model_config)

        # Same model + same benchmarks should produce same scores
        assert result1.overall_score == result2.overall_score
        for r1, r2 in zip(result1.benchmark_results, result2.benchmark_results, strict=True):
            assert r1.score == r2.score

    def test_different_models_produce_different_results(
        self,
        small_suite: BenchmarkSuite,
    ) -> None:
        config = EvaluationConfig(
            models=[
                ModelConfig(
                    name="Cheap",
                    provider="test",
                    model_id="cheap-v1",
                    cost_per_1k_input=0.0001,
                    cost_per_1k_output=0.0002,
                ),
                ModelConfig(
                    name="Expensive",
                    provider="test",
                    model_id="expensive-v1",
                    cost_per_1k_input=0.01,
                    cost_per_1k_output=0.03,
                ),
            ]
        )
        evaluator = ModelEvaluator(config=config, benchmark_suite=small_suite, simulate=True)
        results = evaluator.evaluate_all()

        # Different models should have different scores or latencies
        assert results[0].model_name != results[1].model_name
        # At least latency should differ due to cost-based simulation
        assert results[0].latency_ms != results[1].latency_ms


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
models:
  - name: "Test Model"
    provider: "test"
    model_id: "test-v1"
    cost_per_1k_input: 0.001
    cost_per_1k_output: 0.002
    max_tokens: 2048
    description: "A test model"

evaluation:
  default_num_runs: 5
  timeout_seconds: 60
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)
        assert len(config.models) == 1
        assert config.models[0].name == "Test Model"
        assert config.default_num_runs == 5
        assert config.timeout_seconds == 60

    def test_from_yaml_class_method(self, tmp_path: Path) -> None:
        yaml_content = """
models:
  - name: "YAML Model"
    provider: "test"
    model_id: "yaml-v1"
    cost_per_1k_input: 0.002
    cost_per_1k_output: 0.004

evaluation:
  default_num_runs: 2
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        evaluator = ModelEvaluator.from_yaml(config_file, simulate=True)
        assert len(evaluator.config.models) == 1
        assert evaluator.config.models[0].name == "YAML Model"
        assert evaluator.simulate is True
