#!/usr/bin/env python3
"""
Show actual model responses for a few sample questions.
"""

import os
from api_client import APIClient
from data_loader import load_mmlu_pro_data
from pathlib import Path

# Load API key
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("Error: ANTHROPIC_API_KEY not set")
    exit(1)

# Load questions
data_dir = Path(__file__).parent / "MMLU-Pro" / "data"
questions = load_mmlu_pro_data(data_dir)

# Take first 5 questions
sample_questions = questions[:5]

# System prompt
SYSTEM_PROMPT = (
    "You are taking a multiple choice exam. Read the question carefully and "
    "respond with ONLY the letter of the correct answer (A, B, C, D, E, F, G, H, I, or J). "
    "Do not include any explanation."
)

SCAFFOLDING = "Read carefully. Identify the key concept being tested. Eliminate clearly wrong options first."

client = APIClient(api_key)

print("=" * 100)
print("ACTUAL MODEL RESPONSES - BASELINE vs SCAFFOLDED")
print("=" * 100)

for i, q in enumerate(sample_questions):
    print(f"\n{'='*100}")
    print(f"Question {i+1} - {q.subject.upper()}")
    print(f"{'='*100}")
    print(f"{q.question_text[:200]}...")
    print()
    print(q.format_options()[:300] + "...")
    print(f"\n✓ Correct Answer: {q.correct_answer}")
    
    # Baseline
    baseline_msg = q.question_text + "\n\n" + q.format_options()
    baseline_resp = client.call(SYSTEM_PROMPT, baseline_msg)
    
    # Scaffolded
    scaffolded_msg = SCAFFOLDING + "\n\n" + q.question_text + "\n\n" + q.format_options()
    scaffolded_resp = client.call(SYSTEM_PROMPT, scaffolded_msg)
    
    print(f"\n{'─'*96}")
    print(f"BASELINE Response ({baseline_resp.output_tokens} tokens):")
    print(f"Raw output: \"{baseline_resp.answer_text}\"")
    
    print(f"\n{'─'*96}")
    print(f"SCAFFOLDED Response ({scaffolded_resp.output_tokens} tokens):")
    print(f"Raw output: \"{scaffolded_resp.answer_text}\"")
    print()
