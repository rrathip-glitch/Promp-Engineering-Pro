#!/usr/bin/env python3
"""
Test with LLM fallback extraction - show all outputs.
"""

import os
import random
from api_client import APIClient, extract_answer, extract_answer_with_llm_fallback
from data_loader import load_mmlu_pro_data
from pathlib import Path

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("Error: ANTHROPIC_API_KEY not set")
    exit(1)

data_dir = Path(__file__).parent / "MMLU-Pro" / "data"
questions = load_mmlu_pro_data(data_dir)

random.seed(777)  # New seed for fresh questions
sample_questions = random.sample(questions, 5)

SYSTEM_PROMPT = (
    "You are taking a multiple choice exam. Read the question carefully and "
    "respond with ONLY the letter of the correct answer (A, B, C, D, E, F, G, H, I, or J). "
    "Do not include any explanation."
)

SCAFFOLDING = """[Internal reasoning: Read carefully and expertly preempt likely errors. Note the key concept being tested. Use process of elimination.]

IMPORTANT: Your response must be ONLY a single letter (A-J). No explanation. No other text.

"""

client = APIClient(api_key)

print("=" * 100)
print("LLM FALLBACK EXTRACTION TEST - 20 QUESTIONS")
print("=" * 100)
print()

baseline_results = []
scaffolded_results = []

for i, q in enumerate(sample_questions):
    print(f"Q{i+1:02d} [{q.subject.upper()[:10]:10s}] ", end="", flush=True)
    
    question_with_options = q.question_text + "\n\n" + q.format_options()
    
    # BASELINE
    baseline_resp = client.call(SYSTEM_PROMPT, question_with_options)
    baseline_raw = baseline_resp.answer_text
    baseline_extracted = extract_answer_with_llm_fallback(
        baseline_raw, client, q.question_text, q.format_options()
    )
    baseline_correct = baseline_extracted == q.correct_answer
    baseline_results.append((baseline_raw, baseline_extracted, baseline_correct))
    
    # SCAFFOLDED
    scaffolded_msg = SCAFFOLDING + question_with_options
    scaffolded_resp = client.call(SYSTEM_PROMPT, scaffolded_msg)
    scaffolded_raw = scaffolded_resp.answer_text
    scaffolded_extracted = extract_answer_with_llm_fallback(
        scaffolded_raw, client, q.question_text, q.format_options()
    )
    scaffolded_correct = scaffolded_extracted == q.correct_answer
    scaffolded_results.append((scaffolded_raw, scaffolded_extracted, scaffolded_correct))
    
    # Quick inline status
    b_mark = "✓" if baseline_correct else "✗"
    s_mark = "✓" if scaffolded_correct else "✗"
    print(f"Correct: {q.correct_answer} | Base: {baseline_extracted} {b_mark} | Scaff: {scaffolded_extracted} {s_mark}")

print()
print("=" * 100)
print("DETAILED RESULTS")
print("=" * 100)

for i, q in enumerate(sample_questions):
    b_raw, b_ext, b_ok = baseline_results[i]
    s_raw, s_ext, s_ok = scaffolded_results[i]
    
    print(f"\n--- Q{i+1}: {q.subject.upper()} (Correct: {q.correct_answer}) ---")
    print(f"  BASELINE:   Raw=\"{b_raw[:50]}{'...' if len(b_raw) > 50 else ''}\" → {b_ext} {'✓' if b_ok else '✗'}")
    print(f"  SCAFFOLDED: Raw=\"{s_raw[:50]}{'...' if len(s_raw) > 50 else ''}\" → {s_ext} {'✓' if s_ok else '✗'}")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

baseline_correct_count = sum(1 for _, _, ok in baseline_results if ok)
scaffolded_correct_count = sum(1 for _, _, ok in scaffolded_results if ok)
baseline_valid = sum(1 for _, ext, _ in baseline_results if ext in 'ABCDEFGHIJ')
scaffolded_valid = sum(1 for _, ext, _ in scaffolded_results if ext in 'ABCDEFGHIJ')

print(f"\n  Baseline:   {baseline_correct_count}/20 correct ({baseline_correct_count*5}%)  |  Valid answers: {baseline_valid}/20")
print(f"  Scaffolded: {scaffolded_correct_count}/20 correct ({scaffolded_correct_count*5}%)  |  Valid answers: {scaffolded_valid}/20")
print()

if baseline_valid == 20 and scaffolded_valid == 20:
    print("  ✅ ALL ANSWERS ARE VALID A-J LETTERS!")
else:
    print("  ⚠️  Some answers are still invalid")
print()
