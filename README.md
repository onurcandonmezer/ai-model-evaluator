# AI Model Evaluator

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/onurcandonmezer/ai-model-evaluator/actions/workflows/ci.yml/badge.svg)](https://github.com/onurcandonmezer/ai-model-evaluator/actions)

A comprehensive LLM benchmarking toolkit that systematically compares multiple language models across accuracy, latency, cost, and safety metrics. Run standardized benchmarks, generate detailed comparison reports, and make data-driven decisions about which model to deploy for your use case.

## Key Features

- **Multi-Model Comparison** -- Evaluate GPT-4o, Gemini, Claude, and custom models side by side
- **22 Benchmark Cases** -- Predefined tests across summarization, QA, classification, reasoning, and code generation
- **Comprehensive Metrics** -- Accuracy, F1 score, latency percentiles (P50/P95/P99), cost analysis, hallucination rate, and throughput
- **Interactive Dashboard** -- Streamlit-based UI with charts, tables, and exportable reports
- **Simulation Mode** -- Full evaluation pipeline works without API keys for testing and development
- **YAML Configuration** -- Easily add or modify models through configuration files
- **Markdown/HTML Reports** -- Auto-generated evaluation reports with rankings and recommendations
- **Cost-Quality Analysis** -- Find the optimal cost-performance balance for your needs

## Architecture

```
                    +------------------+
                    |   models.yaml    |  Configuration
                    +--------+---------+
                             |
                    +--------v---------+
                    |  ModelEvaluator   |  Orchestration
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v----------+     +-----------v-----------+
    |   BenchmarkSuite   |     |   Simulated / Live    |
    |  (22 test cases)   |     |     API Responses     |
    +---------+----------+     +-----------+-----------+
              |                             |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |     Metrics      |  Analysis
                    |  Engine (P50,    |
                    |  F1, Cost/QoS)   |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v----------+     +-----------v-----------+
    | ReportGenerator    |     |  Streamlit Dashboard  |
    | (Markdown / HTML)  |     |  (Interactive UI)     |
    +--------------------+     +-----------------------+
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/onurcandonmezer/ai-model-evaluator.git
cd ai-model-evaluator

# Install dependencies
uv venv && uv pip install -e ".[dev]"
```

### Run Evaluation (Simulation Mode)

```python
from src.evaluator import ModelEvaluator

# Load from YAML config and run in simulation mode (no API keys needed)
evaluator = ModelEvaluator.from_yaml("configs/models.yaml", simulate=True)
results = evaluator.evaluate_all()

# Generate comparison report
report = evaluator.get_comparison_report()
print(f"Best Overall: {report.best_overall}")
print(f"Best Accuracy: {report.best_accuracy}")
print(f"Best Latency: {report.best_latency}")
```

### Launch Dashboard

```bash
uv run streamlit run src/app.py
```

## Usage Examples

### Evaluate Specific Models

```python
from src.evaluator import ModelEvaluator, ModelConfig, EvaluationConfig

config = EvaluationConfig(
    models=[
        ModelConfig(
            name="Gemini Flash",
            provider="google",
            model_id="gemini-2.5-flash-lite",
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
        ),
        ModelConfig(
            name="GPT-4o Mini",
            provider="openai",
            model_id="gpt-4o-mini",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
        ),
    ]
)

evaluator = ModelEvaluator(config=config, simulate=True)
results = evaluator.evaluate_all()

for result in results:
    print(f"{result.model_name}: {result.overall_score:.1%} accuracy, "
          f"{result.latency_ms:.0f}ms avg latency")
```

### Generate Reports

```python
from src.report_generator import ReportGenerator

evaluator = ModelEvaluator.from_yaml("configs/models.yaml", simulate=True)
evaluator.evaluate_all()
report = evaluator.get_comparison_report()

generator = ReportGenerator(output_dir="reports")
generator.save_markdown(report, "evaluation_report.md")
generator.save_html(report, "evaluation_report.html")
```

### Custom Benchmarks

```python
from src.benchmarks import BenchmarkSuite, BenchmarkCase, BenchmarkCategory, Difficulty

custom_suite = BenchmarkSuite(cases=[
    BenchmarkCase(
        name="domain_specific_qa",
        category=BenchmarkCategory.QA,
        input_text="What are the key principles of MLOps?",
        expected_output_contains=["monitoring", "deployment", "automation", "reproducibility"],
        difficulty=Difficulty.MEDIUM,
    ),
])

evaluator = ModelEvaluator(config=config, benchmark_suite=custom_suite, simulate=True)
```

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11+ |
| AI APIs | Google Gemini, OpenAI, Anthropic |
| Data Validation | Pydantic |
| Configuration | PyYAML |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| CLI Output | Rich |
| Testing | pytest, pytest-cov |
| Linting | Ruff |
| CI/CD | GitHub Actions |

## Project Structure

```
ai-model-evaluator/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── evaluator.py             # Core evaluation engine
│   ├── benchmarks.py            # Benchmark test suites (22 cases)
│   ├── metrics.py               # Metric calculations & comparison
│   ├── report_generator.py      # Markdown/HTML report generation
│   └── app.py                   # Streamlit dashboard
├── tests/
│   ├── test_evaluator.py        # Evaluator tests (12 tests)
│   ├── test_benchmarks.py       # Benchmark tests (12 tests)
│   └── test_metrics.py          # Metrics tests (19 tests)
├── configs/
│   └── models.yaml              # Model configurations
├── .github/workflows/
│   └── ci.yml                   # CI pipeline
├── pyproject.toml               # Project configuration
├── Makefile                     # Development commands
├── LICENSE                      # MIT License
└── README.md
```

## Development

```bash
# Run tests
make test

# Lint code
make lint

# Format code
make format

# Launch dashboard
make run
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built by <a href="https://github.com/onurcandonmezer">Onurcan Donmezer</a>
</p>
