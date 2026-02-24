"""Streamlit dashboard for interactive LLM evaluation.

Provides a web-based interface for selecting models, running benchmarks,
and visualizing comparison results with interactive charts and tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.benchmarks import BenchmarkCategory, BenchmarkSuite
from src.evaluator import EvaluationConfig, ModelConfig, ModelEvaluator
from src.metrics import format_metrics_table
from src.report_generator import ReportGenerator


def get_default_models() -> list[ModelConfig]:
    """Load default model configurations from YAML or fallback to defaults."""
    config_path = Path(__file__).parent.parent / "configs" / "models.yaml"
    if config_path.exists():
        evaluator = ModelEvaluator.from_yaml(config_path)
        return evaluator.config.models

    return [
        ModelConfig(
            name="Gemini 2.5 Flash Lite",
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
        ModelConfig(
            name="Claude 3 Haiku",
            provider="anthropic",
            model_id="claude-3-haiku-20240307",
            cost_per_1k_input=0.00025,
            cost_per_1k_output=0.00125,
        ),
    ]


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.set_page_config(
        page_title="AI Model Evaluator",
        page_icon="::bar_chart::",
        layout="wide",
    )

    st.title("AI Model Evaluator")
    st.markdown("Systematically compare LLMs across accuracy, latency, cost, and safety metrics.")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    available_models = get_default_models()
    model_names = [m.name for m in available_models]

    selected_models = st.sidebar.multiselect(
        "Select Models to Evaluate",
        options=model_names,
        default=model_names[:3],
    )

    categories = [cat.value for cat in BenchmarkCategory]
    selected_categories = st.sidebar.multiselect(
        "Benchmark Categories",
        options=categories,
        default=categories,
    )

    simulate = st.sidebar.checkbox("Simulation Mode (no API keys required)", value=True)

    # Main area
    if st.sidebar.button("Run Evaluation", type="primary"):
        if not selected_models:
            st.error("Please select at least one model.")
            return

        selected_configs = [m for m in available_models if m.name in selected_models]

        config = EvaluationConfig(models=selected_configs)
        suite = BenchmarkSuite()

        # Filter benchmarks by selected categories
        if selected_categories:
            filtered_cases = [
                c for c in suite.cases
                if c.category.value in selected_categories
            ]
            suite = BenchmarkSuite(cases=filtered_cases)

        evaluator = ModelEvaluator(config=config, benchmark_suite=suite, simulate=simulate)

        with st.spinner("Running evaluation..."):
            results = evaluator.evaluate_all()

        st.success(f"Evaluation complete! Tested {len(results)} models across "
                   f"{suite.total_cases} benchmarks.")

        # Store results in session state
        st.session_state["results"] = results
        st.session_state["evaluator"] = evaluator

    # Display results if available
    if "results" in st.session_state:
        results = st.session_state["results"]
        evaluator = st.session_state["evaluator"]

        # Comparison report
        report = evaluator.get_comparison_report()

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview", "Detailed Metrics", "Category Breakdown", "Report"
        ])

        with tab1:
            st.header("Model Rankings")

            ranking_data = []
            for r in report.rankings:
                ranking_data.append({
                    "Rank": r.rank,
                    "Model": r.model_name,
                    "Overall Score": f"{r.overall_score:.3f}",
                    "Accuracy": f"{r.scores.get('accuracy', 0):.3f}",
                    "Latency": f"{r.scores.get('latency', 0):.3f}",
                    "Cost": f"{r.scores.get('cost', 0):.3f}",
                })

            st.dataframe(pd.DataFrame(ranking_data), use_container_width=True, hide_index=True)

            # Bar chart of overall scores
            st.subheader("Overall Score Comparison")
            chart_data = pd.DataFrame({
                "Model": [r.model_name for r in report.rankings],
                "Score": [r.overall_score for r in report.rankings],
            })
            st.bar_chart(chart_data.set_index("Model"))

            # Best-in-class callouts
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Overall", report.best_overall)
            with col2:
                st.metric("Best Accuracy", report.best_accuracy)
            with col3:
                st.metric("Best Latency", report.best_latency)
            with col4:
                st.metric("Best Cost", report.best_cost)

        with tab2:
            st.header("Detailed Metrics")

            metrics_rows = [format_metrics_table(m) for m in report.models]
            st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

            # Latency comparison
            st.subheader("Latency Comparison (P50, P95, P99)")
            latency_data = pd.DataFrame({
                "Model": [m.model_name for m in report.models],
                "P50": [m.latency.p50 for m in report.models],
                "P95": [m.latency.p95 for m in report.models],
                "P99": [m.latency.p99 for m in report.models],
            })
            st.bar_chart(latency_data.set_index("Model"))

            # Cost comparison
            st.subheader("Cost per 1K Tokens")
            cost_data = pd.DataFrame({
                "Model": [m.model_name for m in report.models],
                "Cost ($)": [m.cost_per_1k_tokens for m in report.models],
            })
            st.bar_chart(cost_data.set_index("Model"))

        with tab3:
            st.header("Category Breakdown")

            all_cats: set[str] = set()
            for model in report.models:
                all_cats.update(model.category_scores.keys())

            if all_cats:
                cat_data: dict[str, list[float]] = {"Category": sorted(all_cats)}
                for model in report.models:
                    cat_data[model.model_name] = [
                        model.category_scores.get(cat, 0.0)
                        for cat in sorted(all_cats)
                    ]

                df = pd.DataFrame(cat_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Category bar chart
                st.bar_chart(df.set_index("Category"))

        with tab4:
            st.header("Generated Report")

            generator = ReportGenerator()
            md_report = generator.generate_markdown(report)
            st.markdown(md_report)

            st.download_button(
                "Download Markdown Report",
                data=md_report,
                file_name="evaluation_report.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
