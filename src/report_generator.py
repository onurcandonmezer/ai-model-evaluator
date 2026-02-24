"""Report generation for LLM evaluation results.

Generates Markdown and HTML reports with comparison tables, rankings,
cost-quality analysis, and executive summaries.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.metrics import (
    ModelComparisonReport,
    format_metrics_table,
)


class ReportGenerator:
    """Generates evaluation reports in Markdown and HTML formats.

    Args:
        output_dir: Directory to write report files to.
    """

    def __init__(self, output_dir: str | Path = "reports") -> None:
        self.output_dir = Path(output_dir)

    def generate_markdown(self, report: ModelComparisonReport) -> str:
        """Generate a complete Markdown evaluation report.

        Args:
            report: ModelComparisonReport with all evaluation data.

        Returns:
            Complete Markdown report as a string.
        """
        sections = [
            self._header(),
            self._executive_summary(report),
            self._ranking_table(report),
            self._detailed_metrics_table(report),
            self._category_breakdown(report),
            self._cost_quality_analysis(report),
            self._best_in_class(report),
            self._footer(),
        ]
        return "\n\n".join(sections)

    def generate_html(self, report: ModelComparisonReport) -> str:
        """Generate an HTML evaluation report.

        Args:
            report: ModelComparisonReport with all evaluation data.

        Returns:
            Complete HTML report as a string.
        """
        markdown_content = self.generate_markdown(report)
        return self._wrap_html(markdown_content)

    def save_markdown(self, report: ModelComparisonReport, filename: str = "report.md") -> Path:
        """Save Markdown report to a file.

        Args:
            report: ModelComparisonReport to save.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / filename
        content = self.generate_markdown(report)
        filepath.write_text(content)
        return filepath

    def save_html(self, report: ModelComparisonReport, filename: str = "report.html") -> Path:
        """Save HTML report to a file.

        Args:
            report: ModelComparisonReport to save.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / filename
        content = self.generate_html(report)
        filepath.write_text(content)
        return filepath

    @staticmethod
    def _header() -> str:
        """Generate report header."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        return f"# AI Model Evaluation Report\n\n**Generated:** {timestamp}\n\n---"

    @staticmethod
    def _executive_summary(report: ModelComparisonReport) -> str:
        """Generate executive summary section."""
        num_models = len(report.models)
        if num_models == 0:
            return "## Executive Summary\n\nNo models were evaluated."

        avg_accuracy = sum(m.accuracy_score for m in report.models) / num_models
        avg_latency = sum(m.latency.p50 for m in report.models) / num_models

        lines = [
            "## Executive Summary",
            "",
            f"This report evaluates **{num_models} models** across multiple benchmark categories.",
            "",
            f"- **Best Overall:** {report.best_overall}",
            f"- **Best Accuracy:** {report.best_accuracy}",
            f"- **Best Latency:** {report.best_latency}",
            f"- **Best Cost Efficiency:** {report.best_cost}",
            "",
            f"Average accuracy across all models: **{avg_accuracy:.1%}**",
            f"Average median latency: **{avg_latency:.0f}ms**",
        ]
        return "\n".join(lines)

    @staticmethod
    def _ranking_table(report: ModelComparisonReport) -> str:
        """Generate model ranking table."""
        lines = [
            "## Model Rankings",
            "",
            "| Rank | Model | Overall Score | Accuracy | Latency | Cost | Hallucination |",
            "|------|-------|--------------|----------|---------|------|---------------|",
        ]

        for r in report.rankings:
            lines.append(
                f"| {r.rank} | {r.model_name} "
                f"| {r.overall_score:.3f} "
                f"| {r.scores.get('accuracy', 0):.3f} "
                f"| {r.scores.get('latency', 0):.3f} "
                f"| {r.scores.get('cost', 0):.3f} "
                f"| {r.scores.get('hallucination', 0):.3f} |"
            )

        return "\n".join(lines)

    @staticmethod
    def _detailed_metrics_table(report: ModelComparisonReport) -> str:
        """Generate detailed metrics comparison table."""
        lines = [
            "## Detailed Metrics",
            "",
            "| Metric | " + " | ".join(m.model_name for m in report.models) + " |",
            "|--------| " + " | ".join("---" for _ in report.models) + " |",
        ]

        if not report.models:
            return "\n".join(lines)

        metrics_data = [format_metrics_table(m) for m in report.models]
        keys = [k for k in metrics_data[0] if k != "Model"]

        for key in keys:
            row = f"| {key} |"
            for md in metrics_data:
                row += f" {md[key]} |"
            lines.append(row)

        return "\n".join(lines)

    @staticmethod
    def _category_breakdown(report: ModelComparisonReport) -> str:
        """Generate per-category score breakdown."""
        lines = ["## Category Breakdown", ""]

        if not report.models:
            lines.append("No category data available.")
            return "\n".join(lines)

        # Collect all categories
        all_categories: set[str] = set()
        for model in report.models:
            all_categories.update(model.category_scores.keys())

        if not all_categories:
            lines.append("No category data available.")
            return "\n".join(lines)

        sorted_categories = sorted(all_categories)
        header = "| Category | " + " | ".join(m.model_name for m in report.models) + " |"
        separator = "|----------| " + " | ".join("---" for _ in report.models) + " |"
        lines.extend([header, separator])

        for cat in sorted_categories:
            row = f"| {cat} |"
            for model in report.models:
                score = model.category_scores.get(cat, 0.0)
                row += f" {score:.1%} |"
            lines.append(row)

        return "\n".join(lines)

    @staticmethod
    def _cost_quality_analysis(report: ModelComparisonReport) -> str:
        """Generate cost-quality scatter description."""
        lines = [
            "## Cost-Quality Analysis",
            "",
            "The following shows each model's position in the cost-quality trade-off space:",
            "",
        ]

        if not report.models:
            lines.append("No data available.")
            return "\n".join(lines)

        for model in report.models:
            quality = model.accuracy_score
            cost = model.cost_per_1k_tokens
            cpq = model.cost_per_quality_score

            cpq_str = "N/A (zero quality)" if cpq == float("inf") else f"${cpq:.4f}"

            lines.append(
                f"- **{model.model_name}**: "
                f"Quality={quality:.1%}, "
                f"Cost=${cost:.4f}/1K tokens, "
                f"Cost-per-Quality={cpq_str}"
            )

        return "\n".join(lines)

    @staticmethod
    def _best_in_class(report: ModelComparisonReport) -> str:
        """Generate best-in-class recommendations."""
        lines = [
            "## Recommendations",
            "",
            "Based on the evaluation results:",
            "",
            f"- **Best for accuracy-critical tasks:** {report.best_accuracy}",
            f"- **Best for latency-sensitive applications:** {report.best_latency}",
            f"- **Best for cost-conscious deployments:** {report.best_cost}",
            f"- **Best overall balanced choice:** {report.best_overall}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _footer() -> str:
        """Generate report footer."""
        return (
            "---\n\n"
            "*Generated by AI Model Evaluator | "
            "[GitHub](https://github.com/onurcandonmezer/ai-model-evaluator)*"
        )

    @staticmethod
    def _wrap_html(markdown_content: str) -> str:
        """Wrap Markdown content in a basic HTML template.

        Args:
            markdown_content: Markdown string to wrap.

        Returns:
            HTML string with embedded Markdown (pre-formatted).
        """
        escaped = markdown_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Model Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
            background: #0d1117;
            color: #c9d1d9;
            line-height: 1.6;
        }}
        pre {{
            background: #161b22;
            padding: 1.5rem;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid #30363d;
            white-space: pre-wrap;
        }}
        h1, h2, h3 {{
            color: #58a6ff;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }}
        th, td {{
            border: 1px solid #30363d;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background: #161b22;
            color: #58a6ff;
        }}
        tr:nth-child(even) {{
            background: #161b22;
        }}
    </style>
</head>
<body>
<pre>{escaped}</pre>
</body>
</html>"""

    def generate_sample_report(self) -> str:
        """Generate a sample report using simulated data for demonstration.

        Returns:
            Markdown string of the sample report.
        """
        from src.evaluator import ModelEvaluator

        config_path = Path(__file__).parent.parent / "configs" / "models.yaml"
        if config_path.exists():
            evaluator = ModelEvaluator.from_yaml(config_path, simulate=True)
        else:
            evaluator = ModelEvaluator(simulate=True)

        evaluator.evaluate_all()
        report = evaluator.get_comparison_report()

        md_content = self.generate_markdown(report)
        print(md_content)
        return md_content
