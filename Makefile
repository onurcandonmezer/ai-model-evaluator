.PHONY: install dev test lint format run clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv pip install -e .

dev: ## Install development dependencies
	uv pip install -e ".[dev]"

test: ## Run tests with coverage
	uv run python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint: ## Run linter checks
	uv run ruff check src/ tests/

format: ## Format code with ruff
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

run: ## Launch Streamlit dashboard
	uv run streamlit run src/app.py

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

report: ## Generate evaluation report
	uv run python -c "from src.report_generator import ReportGenerator; ReportGenerator().generate_sample_report()"
