"""
Evaluator module for running A/B tests.

Handles prompt formatting, test execution, and result aggregation.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from config import ScaffoldingConfig, MODEL_NAME
from data_loader import Question
from api_client import APIClient, APIResponse, extract_answer, extract_answer_with_llm_fallback, BudgetExceededError
from checkpoint import CheckpointManager, TestResult, create_test_result


logger = logging.getLogger(__name__)


# System prompt for multiple choice exams
SYSTEM_PROMPT = (
    "You are taking a multiple choice exam. Read the question carefully and "
    "respond with ONLY the letter of the correct answer (A, B, C, D, E, F, G, H, I, or J). "
    "Do not include any explanation."
)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a condition."""
    condition: str
    total_questions: int
    correct_count: int
    accuracy_pct: float
    total_cost_usd: float
    cost_per_correct_usd: float


def format_user_message(
    question: Question,
    scaffolding: Optional[ScaffoldingConfig] = None,
    include_scaffolding: bool = False
) -> str:
    """
    Format the user message for an API call.
    
    Args:
        question: The question to format
        scaffolding: Scaffolding configuration
        include_scaffolding: Whether to include the pre-prompt
        
    Returns:
        Formatted user message
    """
    parts = []
    
    # Add scaffolding if enabled
    if include_scaffolding and scaffolding and scaffolding.enabled:
        if scaffolding.pre_prompt.strip():
            parts.append(scaffolding.pre_prompt)
            parts.append("")  # Blank line
    
    # Add question and options
    parts.append(question.question_text)
    parts.append("")  # Blank line
    parts.append(question.format_options())
    
    return "\n".join(parts)


def run_single_test(
    client: APIClient,
    question: Question,
    scaffolding: ScaffoldingConfig,
    include_scaffolding: bool,
    model: str = MODEL_NAME
) -> Tuple[str, APIResponse]:
    """
    Run a single test (one API call).
    
    Args:
        client: The API client
        question: The question to test
        scaffolding: Scaffolding configuration
        include_scaffolding: Whether this is the scaffolded condition
        
    Returns:
        Tuple of (extracted_answer, api_response)
    """
    user_message = format_user_message(question, scaffolding, include_scaffolding)
    
    try:
        response = client.call(SYSTEM_PROMPT, user_message, model=model)
        # Use LLM fallback to guarantee valid A-J answer
        extracted = extract_answer_with_llm_fallback(
            response.answer_text,
            client,
            question.question_text,
            question.format_options()
        )
        return extracted, response
        
    except BudgetExceededError:
        raise
    except Exception as e:
        logger.error(f"Error testing question {question.question_id}: {e}")
        raise


def run_question_ab_test(
    client: APIClient,
    question: Question,
    scaffolding: ScaffoldingConfig
) -> Tuple[TestResult, TestResult]:
    """
    Run both baseline and scaffolded tests for a question.
    
    Args:
        client: The API client
        question: The question to test
        scaffolding: Scaffolding configuration
        
    Returns:
        Tuple of (baseline_result, scaffolded_result)
    """
    # Run baseline (no scaffolding)
    baseline_answer, baseline_response = run_single_test(
        client, question, scaffolding, include_scaffolding=False
    )
    
    baseline_result = create_test_result(
        question_id=question.question_id,
        subject=question.subject,
        condition="baseline",
        correct_answer=question.correct_answer,
        model_answer=baseline_answer,
        input_tokens=baseline_response.input_tokens,
        output_tokens=baseline_response.output_tokens,
        latency_sec=baseline_response.latency_seconds,
        cost_usd=baseline_response.cost_usd
    )
    
    # Run scaffolded
    scaffolded_answer, scaffolded_response = run_single_test(
        client, question, scaffolding, include_scaffolding=True
    )
    
    scaffolded_result = create_test_result(
        question_id=question.question_id,
        subject=question.subject,
        condition="scaffolded",
        correct_answer=question.correct_answer,
        model_answer=scaffolded_answer,
        input_tokens=scaffolded_response.input_tokens,
        output_tokens=scaffolded_response.output_tokens,
        latency_sec=scaffolded_response.latency_seconds,
        cost_usd=scaffolded_response.cost_usd
    )
    
    return baseline_result, scaffolded_result


def aggregate_results(results: List[TestResult]) -> Dict[str, AggregatedMetrics]:
    """
    Aggregate results by condition.
    
    Returns dict mapping condition name to aggregated metrics.
    """
    by_condition: Dict[str, List[TestResult]] = {}
    
    for result in results:
        if result.condition not in by_condition:
            by_condition[result.condition] = []
        by_condition[result.condition].append(result)
    
    aggregated = {}
    
    for condition, cond_results in by_condition.items():
        total = len(cond_results)
        correct = sum(1 for r in cond_results if r.is_correct)
        total_cost = sum(r.cost_usd for r in cond_results)
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        cost_per_correct = (total_cost / correct) if correct > 0 else float('inf')
        
        aggregated[condition] = AggregatedMetrics(
            condition=condition,
            total_questions=total,
            correct_count=correct,
            accuracy_pct=accuracy,
            total_cost_usd=total_cost,
            cost_per_correct_usd=cost_per_correct
        )
    
    return aggregated


def write_detailed_csv(results: List[TestResult], output_path: Path) -> None:
    """Write detailed results to CSV file."""
    fieldnames = [
        'question_id', 'subject', 'condition', 'correct_answer',
        'model_answer', 'is_correct', 'input_tokens', 'output_tokens',
        'latency_sec', 'cost_usd'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'question_id': result.question_id,
                'subject': result.subject,
                'condition': result.condition,
                'correct_answer': result.correct_answer,
                'model_answer': result.model_answer,
                'is_correct': 1 if result.is_correct else 0,
                'input_tokens': result.input_tokens,
                'output_tokens': result.output_tokens,
                'latency_sec': f"{result.latency_sec:.3f}",
                'cost_usd': f"{result.cost_usd:.6f}"
            })
    
    logger.info(f"Wrote {len(results)} results to {output_path}")


def write_summary_csv(aggregated: Dict[str, AggregatedMetrics], output_path: Path) -> None:
    """Write summary metrics to CSV file."""
    fieldnames = [
        'condition', 'total_questions', 'correct_count',
        'accuracy_pct', 'total_cost_usd', 'cost_per_correct_usd'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for metrics in aggregated.values():
            writer.writerow({
                'condition': metrics.condition,
                'total_questions': metrics.total_questions,
                'correct_count': metrics.correct_count,
                'accuracy_pct': f"{metrics.accuracy_pct:.1f}",
                'total_cost_usd': f"{metrics.total_cost_usd:.4f}",
                'cost_per_correct_usd': f"{metrics.cost_per_correct_usd:.4f}" 
                    if metrics.cost_per_correct_usd != float('inf') else "N/A"
            })
    
    logger.info(f"Wrote summary to {output_path}")


def print_summary(aggregated: Dict[str, AggregatedMetrics]) -> None:
    """Print formatted summary to console."""
    print("\n" + "=" * 40)
    print("=== MMLU-Pro A/B Test Results ===")
    print("=" * 40 + "\n")
    
    baseline = aggregated.get('baseline')
    scaffolded = aggregated.get('scaffolded')
    
    if baseline:
        print("Baseline:")
        print(f"  Accuracy: {baseline.accuracy_pct:.1f}% ({baseline.correct_count}/{baseline.total_questions} correct)")
        print(f"  Total Cost: ${baseline.total_cost_usd:.4f}")
        if baseline.cost_per_correct_usd != float('inf'):
            print(f"  Cost per Correct Answer: ${baseline.cost_per_correct_usd:.4f}")
        else:
            print("  Cost per Correct Answer: N/A (no correct answers)")
        print()
    
    if scaffolded:
        print("Scaffolded:")
        print(f"  Accuracy: {scaffolded.accuracy_pct:.1f}% ({scaffolded.correct_count}/{scaffolded.total_questions} correct)")
        print(f"  Total Cost: ${scaffolded.total_cost_usd:.4f}")
        if scaffolded.cost_per_correct_usd != float('inf'):
            print(f"  Cost per Correct Answer: ${scaffolded.cost_per_correct_usd:.4f}")
        else:
            print("  Cost per Correct Answer: N/A (no correct answers)")
        print()
    
    # Calculate deltas if both conditions present
    if baseline and scaffolded:
        accuracy_delta = scaffolded.accuracy_pct - baseline.accuracy_pct
        cost_delta = scaffolded.total_cost_usd - baseline.total_cost_usd
        
        accuracy_sign = "+" if accuracy_delta >= 0 else ""
        cost_sign = "+" if cost_delta >= 0 else ""
        
        print("Delta (Scaffolded vs Baseline):")
        print(f"  Accuracy: {accuracy_sign}{accuracy_delta:.1f}%")
        print(f"  Total Cost: {cost_sign}${cost_delta:.4f}")
        
        if (baseline.cost_per_correct_usd != float('inf') and 
            scaffolded.cost_per_correct_usd != float('inf')):
            efficiency_delta = scaffolded.cost_per_correct_usd - baseline.cost_per_correct_usd
            eff_sign = "+" if efficiency_delta >= 0 else ""
            print(f"  Cost Efficiency: {eff_sign}${efficiency_delta:.4f} per correct")
    
    print("\n" + "=" * 40)
