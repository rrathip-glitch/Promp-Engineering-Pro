#!/usr/bin/env python3
"""
MMLU-Pro A/B Testing Harness for Claude Haiku 4.5

Main entry point for running baseline vs scaffolded evaluation.

Usage:
    python main.py [--sample-size N] [--dry-run] [--data-dir PATH] [--clear-checkpoint]
"""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from config import load_config, BUDGET_CEILING_USD, RANDOM_SEED, DEFAULT_SAMPLE_SIZE
from data_loader import load_and_sample, Question
from api_client import APIClient, BudgetExceededError
from checkpoint import CheckpointManager
from evaluator import (
    run_question_ab_test,
    aggregate_results,
    write_detailed_csv,
    write_summary_csv,
    print_summary
)
from report_generator import generate_html_report


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MMLU-Pro A/B Testing Harness for Claude Haiku 4.5"
    )
    
    parser.add_argument(
        "--sample-size", "-n",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of questions to sample (default: {DEFAULT_SAMPLE_SIZE})"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate API calls without actually calling (for testing)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to MMLU-Pro data directory (default: ./MMLU-Pro/data)"
    )
    
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear existing checkpoint and start fresh"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def find_data_directory() -> Path:
    """Find the MMLU-Pro data directory."""
    # Check common locations
    candidates = [
        Path(__file__).parent / "MMLU-Pro" / "data",
        Path(__file__).parent / "MMLU-Pro",
        Path(__file__).parent / "data",
        Path.cwd() / "MMLU-Pro" / "data",
        Path.cwd() / "MMLU-Pro",
        Path.cwd() / "data",
    ]
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Check if it has data files
            has_files = any(
                candidate.glob("*.parquet") or 
                candidate.glob("*.json") or
                candidate.glob("*.jsonl")
            )
            if has_files:
                return candidate
            # Check subdirectories
            for subdir in candidate.iterdir():
                if subdir.is_dir():
                    has_files = any(
                        subdir.glob("*.parquet") or 
                        subdir.glob("*.json") or
                        subdir.glob("*.jsonl")
                    )
                    if has_files:
                        return candidate
    
    return None


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up paths
    project_dir = Path(__file__).parent
    checkpoint_path = project_dir / "checkpoint.jsonl"
    detailed_csv_path = project_dir / "results_detailed.csv"
    summary_csv_path = project_dir / "results_summary.csv"
    
    # Load configuration
    print("\n📋 Loading configuration...")
    try:
        config = load_config(sample_size=args.sample_size, dry_run=args.dry_run)
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return 1
    
    if args.dry_run:
        print("🧪 DRY RUN MODE - No actual API calls will be made")
    
    print(f"✓ API key loaded")
    print(f"✓ Scaffolding: {'enabled' if config.scaffolding.enabled else 'disabled'}")
    if config.scaffolding.enabled:
        preview = config.scaffolding.pre_prompt[:50]
        if len(config.scaffolding.pre_prompt) > 50:
            preview += "..."
        print(f"  Pre-prompt: \"{preview}\"")
    
    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = find_data_directory()
    
    if not data_dir or not data_dir.exists():
        print("\n❌ MMLU-Pro data directory not found!")
        print("\nPlease download the dataset:")
        print("  git clone https://github.com/TIGER-AI-Lab/MMLU-Pro.git")
        print("\nOr specify the data directory:")
        print("  python main.py --data-dir /path/to/MMLU-Pro/data")
        return 1
    
    print(f"\n📁 Data directory: {data_dir}")
    
    # Load and sample questions
    print(f"\n📊 Loading and sampling {args.sample_size} questions (seed={RANDOM_SEED})...")
    try:
        questions = load_and_sample(data_dir, args.sample_size, RANDOM_SEED)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        logger.exception("Data loading failed")
        return 1
    
    if not questions:
        print("❌ No questions loaded!")
        return 1
    
    # Get unique subjects
    subjects = set(q.subject for q in questions)
    print(f"✓ Loaded {len(questions)} questions from {len(subjects)} subjects")
    
    # Initialize checkpoint manager
    checkpoint = CheckpointManager(checkpoint_path)
    
    if args.clear_checkpoint:
        print("\n🗑️  Clearing checkpoint...")
        checkpoint.clear()
    
    completed_ids = checkpoint.get_completed_ids()
    if completed_ids:
        print(f"\n📌 Resuming: {len(completed_ids)} questions already completed")
    
    # Filter out completed questions
    remaining_questions = [q for q in questions if q.question_id not in completed_ids]
    
    if not remaining_questions:
        print("\n✅ All questions already completed!")
        # Just generate reports from checkpoint
        results = checkpoint.get_all_results()
        aggregated = aggregate_results(results)
        write_detailed_csv(results, detailed_csv_path)
        write_summary_csv(aggregated, summary_csv_path)
        print_summary(aggregated)
        return 0
    
    print(f"📝 {len(remaining_questions)} questions to process\n")
    
    # Initialize API client
    client = APIClient(config.api_key, dry_run=args.dry_run)
    
    # Restore cumulative cost from checkpoint
    prior_cost = checkpoint.get_cumulative_cost()
    if prior_cost > 0:
        client.cumulative_cost = prior_cost
        print(f"💰 Prior spending from checkpoint: ${prior_cost:.4f}")
    
    print(f"💰 Budget ceiling: ${BUDGET_CEILING_USD:.2f}\n")
    
    # Run tests with progress bar
    budget_exceeded = False
    questions_processed = 0
    questions_failed = 0
    
    try:
        with tqdm(remaining_questions, desc="Testing", unit="question") as pbar:
            for question in pbar:
                try:
                    # Update progress bar description
                    pbar.set_postfix({
                        'cost': f"${client.cumulative_cost:.3f}",
                        'subject': question.subject[:15]
                    })
                    
                    # Run baseline and scaffolded tests
                    baseline_result, scaffolded_result = run_question_ab_test(
                        client, question, config.scaffolding
                    )
                    
                    # Save checkpoint
                    checkpoint.save_question(
                        question.question_id,
                        baseline_result,
                        scaffolded_result
                    )
                    
                    questions_processed += 1
                    
                except BudgetExceededError as e:
                    print(f"\n\n⚠️  {e}")
                    budget_exceeded = True
                    break
                    
                except Exception as e:
                    logger.error(f"Failed on question {question.question_id}: {e}")
                    questions_failed += 1
                    # Continue with next question
                    continue
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Progress has been saved to checkpoint.")
    
    # Generate final reports
    print("\n\n📊 Generating reports...")
    
    results = checkpoint.get_all_results()
    
    if not results:
        print("❌ No results to report!")
        return 1
    
    aggregated = aggregate_results(results)
    
    write_detailed_csv(results, detailed_csv_path)
    write_summary_csv(aggregated, summary_csv_path)
    
    print(f"✓ Detailed results: {detailed_csv_path}")
    print(f"✓ Summary: {summary_csv_path}")
    
    # Generate HTML report
    report_path = project_dir / "results_report.html"
    baseline = aggregated.get('baseline')
    scaffolded = aggregated.get('scaffolded')
    if baseline and scaffolded:
        generate_html_report(
            baseline_accuracy=baseline.accuracy_pct,
            baseline_correct=baseline.correct_count,
            baseline_total=baseline.total_questions,
            baseline_cost=baseline.total_cost_usd,
            baseline_cost_per_correct=baseline.cost_per_correct_usd,
            scaffolded_accuracy=scaffolded.accuracy_pct,
            scaffolded_correct=scaffolded.correct_count,
            scaffolded_total=scaffolded.total_questions,
            scaffolded_cost=scaffolded.total_cost_usd,
            scaffolded_cost_per_correct=scaffolded.cost_per_correct_usd,
            scaffolding_prompt=config.scaffolding.pre_prompt,
            output_path=report_path
        )
        print(f"✓ HTML Report: {report_path}")
    
    # Print console summary
    print_summary(aggregated)
    
    # Print run statistics
    total_cost, total_calls = client.get_stats()
    print(f"\n📈 Run Statistics:")
    print(f"  Questions processed this run: {questions_processed}")
    print(f"  Questions failed: {questions_failed}")
    print(f"  API calls this run: {total_calls}")
    print(f"  Total cost (all runs): ${checkpoint.get_cumulative_cost():.4f}")
    
    if budget_exceeded:
        print("\n⚠️  Run stopped due to budget ceiling")
        return 2
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
