"""Benchmark test suites for LLM evaluation.

Provides predefined benchmark cases across multiple categories including
summarization, question-answering, classification, reasoning, and code generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class BenchmarkCategory(StrEnum):
    """Categories of benchmark tests."""

    SUMMARIZATION = "summarization"
    QA = "qa"
    CLASSIFICATION = "classification"
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"


class Difficulty(StrEnum):
    """Difficulty levels for benchmark cases."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True)
class BenchmarkCase:
    """A single benchmark test case.

    Attributes:
        name: Human-readable name for the benchmark.
        category: The category this benchmark belongs to.
        input_text: The prompt or input to send to the model.
        expected_output_contains: Keywords or phrases expected in the output.
        difficulty: Difficulty level of the benchmark.
        max_tokens: Maximum tokens for the model response.
    """

    name: str
    category: BenchmarkCategory
    input_text: str
    expected_output_contains: list[str]
    difficulty: Difficulty = Difficulty.MEDIUM
    max_tokens: int = 512


@dataclass
class BenchmarkResult:
    """Result from running a single benchmark case.

    Attributes:
        benchmark_name: Name of the benchmark case.
        category: Category of the benchmark.
        model_output: The raw model output text.
        expected_keywords: Keywords that were expected.
        found_keywords: Keywords that were actually found in the output.
        latency_ms: Time taken for the model to respond in milliseconds.
        token_count: Number of tokens in the response.
        passed: Whether the benchmark was considered passed.
        score: Numeric score from 0.0 to 1.0.
    """

    benchmark_name: str
    category: BenchmarkCategory
    model_output: str
    expected_keywords: list[str]
    found_keywords: list[str]
    latency_ms: float
    token_count: int
    passed: bool
    score: float


@dataclass
class BenchmarkSuite:
    """Collection of benchmark test cases for evaluating LLMs.

    Provides a curated set of benchmark cases across multiple categories
    to systematically evaluate model capabilities.
    """

    cases: list[BenchmarkCase] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.cases:
            self.cases = self._get_default_cases()

    def get_by_category(self, category: BenchmarkCategory) -> list[BenchmarkCase]:
        """Filter benchmark cases by category.

        Args:
            category: The category to filter by.

        Returns:
            List of benchmark cases matching the category.
        """
        return [c for c in self.cases if c.category == category]

    def get_by_difficulty(self, difficulty: Difficulty) -> list[BenchmarkCase]:
        """Filter benchmark cases by difficulty level.

        Args:
            difficulty: The difficulty level to filter by.

        Returns:
            List of benchmark cases matching the difficulty.
        """
        return [c for c in self.cases if c.difficulty == difficulty]

    @property
    def categories(self) -> list[BenchmarkCategory]:
        """Get all unique categories present in the suite."""
        return list(set(c.category for c in self.cases))

    @property
    def total_cases(self) -> int:
        """Get total number of benchmark cases."""
        return len(self.cases)

    @staticmethod
    def _get_default_cases() -> list[BenchmarkCase]:
        """Generate the default set of benchmark cases."""
        return [
            # --- Summarization (4 cases) ---
            BenchmarkCase(
                name="news_article_summary",
                category=BenchmarkCategory.SUMMARIZATION,
                input_text=(
                    "Summarize the following: Artificial intelligence researchers have developed "
                    "a new technique for training large language models that reduces computational "
                    "costs by 40% while maintaining performance. The method, called efficient "
                    "attention pruning, selectively removes redundant attention heads during "
                    "training. Published in Nature Machine Intelligence, the study shows "
                    "promising results across multiple benchmarks."
                ),
                expected_output_contains=["training", "language models", "cost", "attention"],
                difficulty=Difficulty.EASY,
            ),
            BenchmarkCase(
                name="technical_paper_summary",
                category=BenchmarkCategory.SUMMARIZATION,
                input_text=(
                    "Summarize this technical content: Transformer architectures rely on "
                    "self-attention mechanisms that compute pairwise interactions between all "
                    "tokens in a sequence. This quadratic complexity O(n^2) limits the maximum "
                    "context length. Recent work on linear attention, sparse attention patterns, "
                    "and sliding window approaches aim to reduce this to O(n) or O(n log n), "
                    "enabling processing of documents with millions of tokens."
                ),
                expected_output_contains=["attention", "complexity", "tokens", "transformer"],
                difficulty=Difficulty.MEDIUM,
            ),
            BenchmarkCase(
                name="meeting_notes_summary",
                category=BenchmarkCategory.SUMMARIZATION,
                input_text=(
                    "Summarize these meeting notes: Q3 product review meeting. Attendees: "
                    "engineering, product, design teams. Key decisions: 1) Launch beta for "
                    "AI assistant feature by October 15. 2) Increase test coverage to 90%. "
                    "3) Hire 2 ML engineers. Action items: Sarah to draft requirements doc, "
                    "Mike to set up CI/CD pipeline, Lisa to post job descriptions."
                ),
                expected_output_contains=["beta", "October", "coverage", "action"],
                difficulty=Difficulty.EASY,
            ),
            BenchmarkCase(
                name="research_findings_summary",
                category=BenchmarkCategory.SUMMARIZATION,
                input_text=(
                    "Summarize: A longitudinal study of 10,000 participants over 5 years found "
                    "that regular physical exercise (minimum 150 minutes per week) reduced the "
                    "risk of cognitive decline by 35%. The effect was strongest in participants "
                    "aged 50-65. Combining exercise with cognitive training games improved "
                    "outcomes by an additional 15%."
                ),
                expected_output_contains=["exercise", "cognitive", "decline", "participants"],
                difficulty=Difficulty.EASY,
            ),
            # --- Question Answering (5 cases) ---
            BenchmarkCase(
                name="factual_qa_science",
                category=BenchmarkCategory.QA,
                input_text="What is photosynthesis and why is it important for life on Earth?",
                expected_output_contains=["sunlight", "oxygen", "carbon dioxide", "energy"],
                difficulty=Difficulty.EASY,
            ),
            BenchmarkCase(
                name="factual_qa_technology",
                category=BenchmarkCategory.QA,
                input_text=(
                    "Explain the difference between supervised and unsupervised machine learning."
                ),
                expected_output_contains=["labeled", "data", "patterns", "training"],
                difficulty=Difficulty.MEDIUM,
            ),
            BenchmarkCase(
                name="contextual_qa",
                category=BenchmarkCategory.QA,
                input_text=(
                    "Context: The Python programming language was created by Guido van Rossum "
                    "and first released in 1991. It emphasizes code readability with its use "
                    "of significant indentation. Question: Who created Python and when?"
                ),
                expected_output_contains=["Guido van Rossum", "1991"],
                difficulty=Difficulty.EASY,
            ),
            BenchmarkCase(
                name="multi_hop_qa",
                category=BenchmarkCategory.QA,
                input_text=(
                    "The Eiffel Tower is in Paris. Paris is the capital of France. France is "
                    "in Europe. What continent is the Eiffel Tower located on?"
                ),
                expected_output_contains=["Europe"],
                difficulty=Difficulty.MEDIUM,
            ),
            BenchmarkCase(
                name="complex_qa_reasoning",
                category=BenchmarkCategory.QA,
                input_text=(
                    "If a train travels at 120 km/h for 2.5 hours, then slows to 80 km/h "
                    "for 1.5 hours, what is the total distance traveled and the average speed?"
                ),
                expected_output_contains=["420", "105"],
                difficulty=Difficulty.HARD,
            ),
            # --- Classification (4 cases) ---
            BenchmarkCase(
                name="sentiment_positive",
                category=BenchmarkCategory.CLASSIFICATION,
                input_text=(
                    "Classify the sentiment of this review as positive, negative, or neutral: "
                    "'This product exceeded all my expectations! The build quality is "
                    "outstanding and the customer service was incredibly helpful.'"
                ),
                expected_output_contains=["positive"],
                difficulty=Difficulty.EASY,
            ),
            BenchmarkCase(
                name="sentiment_negative",
                category=BenchmarkCategory.CLASSIFICATION,
                input_text=(
                    "Classify the sentiment: 'Terrible experience. The product broke after "
                    "two days and the company refused to issue a refund. Would not recommend.'"
                ),
                expected_output_contains=["negative"],
                difficulty=Difficulty.EASY,
            ),
            BenchmarkCase(
                name="topic_classification",
                category=BenchmarkCategory.CLASSIFICATION,
                input_text=(
                    "Classify this text into one of: technology, sports, politics, science. "
                    "Text: 'The new quantum processor achieved a breakthrough in error "
                    "correction, bringing fault-tolerant quantum computing closer to reality.'"
                ),
                expected_output_contains=["technology", "science"],
                difficulty=Difficulty.MEDIUM,
            ),
            BenchmarkCase(
                name="intent_classification",
                category=BenchmarkCategory.CLASSIFICATION,
                input_text=(
                    "Classify the user intent as: complaint, inquiry, feedback, or request. "
                    "User message: 'I need to update my shipping address for order #12345. "
                    "Can you help me with that?'"
                ),
                expected_output_contains=["request"],
                difficulty=Difficulty.MEDIUM,
            ),
            # --- Reasoning (4 cases) ---
            BenchmarkCase(
                name="logical_deduction",
                category=BenchmarkCategory.REASONING,
                input_text=(
                    "All roses are flowers. Some flowers fade quickly. Can we conclude that "
                    "some roses fade quickly? Explain your reasoning."
                ),
                expected_output_contains=["cannot", "conclude", "not necessarily"],
                difficulty=Difficulty.MEDIUM,
            ),
            BenchmarkCase(
                name="math_reasoning",
                category=BenchmarkCategory.REASONING,
                input_text=(
                    "A store offers a 20% discount on a $80 item, then applies an additional "
                    "10% discount on the reduced price. What is the final price?"
                ),
                expected_output_contains=["57.60"],
                difficulty=Difficulty.MEDIUM,
            ),
            BenchmarkCase(
                name="causal_reasoning",
                category=BenchmarkCategory.REASONING,
                input_text=(
                    "Ice cream sales and drowning incidents both increase during summer. "
                    "Does this mean ice cream causes drowning? Explain the logical fallacy."
                ),
                expected_output_contains=["correlation", "causation"],
                difficulty=Difficulty.MEDIUM,
            ),
            BenchmarkCase(
                name="abstract_reasoning",
                category=BenchmarkCategory.REASONING,
                input_text=(
                    "In a sequence: 2, 6, 12, 20, 30, ... What is the next number? "
                    "Explain the pattern."
                ),
                expected_output_contains=["42"],
                difficulty=Difficulty.HARD,
            ),
            # --- Code Generation (5 cases) ---
            BenchmarkCase(
                name="python_function",
                category=BenchmarkCategory.CODE_GENERATION,
                input_text=(
                    "Write a Python function called 'fibonacci' that takes an integer n "
                    "and returns the nth Fibonacci number using iteration."
                ),
                expected_output_contains=["def fibonacci", "for", "return"],
                difficulty=Difficulty.EASY,
            ),
            BenchmarkCase(
                name="python_class",
                category=BenchmarkCategory.CODE_GENERATION,
                input_text=(
                    "Write a Python class called 'Stack' with push, pop, peek, and is_empty "
                    "methods. Include type hints."
                ),
                expected_output_contains=["class Stack", "def push", "def pop", "def peek"],
                difficulty=Difficulty.MEDIUM,
            ),
            BenchmarkCase(
                name="sql_query",
                category=BenchmarkCategory.CODE_GENERATION,
                input_text=(
                    "Write a SQL query to find the top 5 customers by total order amount "
                    "from tables 'customers' (id, name) and 'orders' (id, customer_id, amount)."
                ),
                expected_output_contains=["SELECT", "JOIN", "ORDER BY", "LIMIT"],
                difficulty=Difficulty.MEDIUM,
            ),
            BenchmarkCase(
                name="algorithm_implementation",
                category=BenchmarkCategory.CODE_GENERATION,
                input_text=(
                    "Implement binary search in Python. The function should take a sorted "
                    "list and a target value, returning the index if found or -1 if not found."
                ),
                expected_output_contains=["def", "binary", "mid", "return"],
                difficulty=Difficulty.MEDIUM,
            ),
            BenchmarkCase(
                name="async_code",
                category=BenchmarkCategory.CODE_GENERATION,
                input_text=(
                    "Write an async Python function that fetches data from multiple URLs "
                    "concurrently using aiohttp and asyncio.gather."
                ),
                expected_output_contains=["async def", "await", "gather", "aiohttp"],
                difficulty=Difficulty.HARD,
            ),
        ]

    def evaluate_output(
        self,
        benchmark: BenchmarkCase,
        model_output: str,
        latency_ms: float,
        token_count: int,
    ) -> BenchmarkResult:
        """Evaluate model output against a benchmark case.

        Args:
            benchmark: The benchmark case to evaluate against.
            model_output: The raw model output text.
            latency_ms: Time taken in milliseconds.
            token_count: Number of tokens in the response.

        Returns:
            BenchmarkResult with scoring details.
        """
        output_lower = model_output.lower()
        found_keywords = [
            kw
            for kw in benchmark.expected_output_contains
            if kw.lower() in output_lower
        ]

        total_expected = len(benchmark.expected_output_contains)
        score = len(found_keywords) / total_expected if total_expected > 0 else 0.0
        passed = score >= 0.5

        return BenchmarkResult(
            benchmark_name=benchmark.name,
            category=benchmark.category,
            model_output=model_output,
            expected_keywords=benchmark.expected_output_contains,
            found_keywords=found_keywords,
            latency_ms=latency_ms,
            token_count=token_count,
            passed=passed,
            score=score,
        )
