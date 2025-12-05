#!/usr/bin/env python3
"""
Test different scaffolding strategies to find one that doesn't break output format.
"""

import os
from api_client import APIClient
from data_loader import load_mmlu_pro_data
from pathlib import Path
import random

# Load API key
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("Error: ANTHROPIC_API_KEY not set")
    exit(1)

# Load questions
data_dir = Path(__file__).parent / "MMLU-Pro" / "data"
questions = load_mmlu_pro_data(data_dir)

# Random sample of 5 questions from different subjects
random.seed(99)  # Different seed for new questions
sample_questions = random.sample(questions, 5)

# System prompt (same for both)
SYSTEM_PROMPT = (
    "You are taking a multiple choice exam. Read the question carefully and "
    "respond with ONLY the letter of the correct answer (A, B, C, D, E, F, G, H, I, or J). "
    "Do not include any explanation."
)

# OLD scaffolding (broken - triggers explanation mode)
OLD_SCAFFOLDING = "Read carefully. Identify the key concept being tested. Eliminate clearly wrong options first."

# NEW sophisticated scaffolding strategies to test
# Strategy: Reinforce output constraint AFTER scaffolding hint
NEW_SCAFFOLDING = """[Internal reasoning: Focus on the key concept. Eliminate obviously wrong options.]

IMPORTANT: Your response must be ONLY a single letter (A-J). No other text.

"""

client = APIClient(api_key)

print("=" * 100)
print("TESTING NEW SCAFFOLDING STRATEGY")
print("=" * 100)
print()
print("OLD scaffolding (broken):")
print(f'  "{OLD_SCAFFOLDING}"')
print()
print("NEW scaffolding (with output reinforcement):")
print(f'  "{NEW_SCAFFOLDING.strip()}"')
print()

baseline_correct = 0
old_scaffolded_correct = 0
new_scaffolded_correct = 0

for i, q in enumerate(sample_questions):
    print(f"\n{'='*100}")
    print(f"Question {i+1} - {q.subject.upper()}")
    print(f"{'='*100}")
    print(f"{q.question_text[:180]}...")
    print()
    
    # Show first 4 options
    letters = "ABCDEFGHIJ"
    for j, opt in enumerate(q.options[:4]):
        marker = "→" if letters[j] == q.correct_answer else " "
        opt_str = str(opt)[:55] + "..." if len(str(opt)) > 55 else str(opt)
        print(f"  {marker} {letters[j]}. {opt_str}")
    if len(q.options) > 4:
        print(f"  ... ({len(q.options)-4} more options)")
    
    print(f"\n  ✓ Correct Answer: {q.correct_answer}")
    
    # Format question
    question_with_options = q.question_text + "\n\n" + q.format_options()
    
    # 1. BASELINE
    baseline_msg = question_with_options
    baseline_resp = client.call(SYSTEM_PROMPT, baseline_msg)
    baseline_is_correct = baseline_resp.answer_text.strip().upper() == q.correct_answer
    if baseline_is_correct:
        baseline_correct += 1
    
    # 2. OLD SCAFFOLDING
    old_scaffolded_msg = OLD_SCAFFOLDING + "\n\n" + question_with_options
    old_scaffolded_resp = client.call(SYSTEM_PROMPT, old_scaffolded_msg)
    old_is_correct = old_scaffolded_resp.answer_text.strip().upper()[:1] == q.correct_answer
    if old_is_correct:
        old_scaffolded_correct += 1
    
    # 3. NEW SCAFFOLDING
    new_scaffolded_msg = NEW_SCAFFOLDING + question_with_options
    new_scaffolded_resp = client.call(SYSTEM_PROMPT, new_scaffolded_msg)
    new_is_correct = new_scaffolded_resp.answer_text.strip().upper()[:1] == q.correct_answer
    if new_is_correct:
        new_scaffolded_correct += 1
    
    print(f"\n  {'─'*94}")
    print(f"  BASELINE ({baseline_resp.output_tokens} tokens):")
    print(f'  Raw: "{baseline_resp.answer_text}" {"✓" if baseline_is_correct else "✗"}')
    
    print(f"\n  {'─'*94}")
    print(f"  OLD SCAFFOLDING ({old_scaffolded_resp.output_tokens} tokens):")
    print(f'  Raw: "{old_scaffolded_resp.answer_text}" {"✓" if old_is_correct else "✗"}')
    
    print(f"\n  {'─'*94}")
    print(f"  NEW SCAFFOLDING ({new_scaffolded_resp.output_tokens} tokens):")
    print(f'  Raw: "{new_scaffolded_resp.answer_text}" {"✓" if new_is_correct else "✗"}')

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"  Baseline:        {baseline_correct}/5 correct ({baseline_correct*20}%)")
print(f"  Old Scaffolding: {old_scaffolded_correct}/5 correct ({old_scaffolded_correct*20}%)")
print(f"  New Scaffolding: {new_scaffolded_correct}/5 correct ({new_scaffolded_correct*20}%)")
print()
