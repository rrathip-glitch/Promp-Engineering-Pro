"""
Checkpoint module for save/resume functionality.

Saves progress after each question to allow resuming interrupted runs.
"""

import json
from pathlib import Path
from typing import Set, List, Dict, Any
from dataclasses import dataclass, asdict
import logging


logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test (one API call)."""
    question_id: str
    subject: str
    condition: str  # 'baseline' or 'scaffolded'
    correct_answer: str
    model_answer: str
    is_correct: bool
    input_tokens: int
    output_tokens: int
    latency_sec: float
    cost_usd: float


@dataclass
class QuestionCheckpoint:
    """Checkpoint for a completed question (both conditions)."""
    question_id: str
    baseline_result: TestResult
    scaffolded_result: TestResult


class CheckpointManager:
    """
    Manages checkpoint save/load for resume capability.
    
    Uses JSONL format for append-friendly storage.
    """
    
    def __init__(self, checkpoint_path: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_path: Path to the checkpoint JSONL file
        """
        self.checkpoint_path = checkpoint_path
        self._completed_ids: Set[str] = set()
        self._results: List[TestResult] = []
        self._load_existing()
    
    def _load_existing(self) -> None:
        """Load existing checkpoint data if file exists."""
        if not self.checkpoint_path.exists():
            logger.info("No existing checkpoint found, starting fresh")
            return
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    question_id = data.get('question_id')
                    
                    if question_id:
                        self._completed_ids.add(question_id)
                        
                        # Reconstruct results
                        if 'baseline_result' in data:
                            self._results.append(
                                TestResult(**data['baseline_result'])
                            )
                        if 'scaffolded_result' in data:
                            self._results.append(
                                TestResult(**data['scaffolded_result'])
                            )
            
            logger.info(
                f"Loaded checkpoint: {len(self._completed_ids)} questions completed, "
                f"{len(self._results)} total results"
            )
            
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}")
            logger.info("Starting fresh due to checkpoint load error")
            self._completed_ids = set()
            self._results = []
    
    def is_completed(self, question_id: str) -> bool:
        """Check if a question has already been completed."""
        return question_id in self._completed_ids
    
    def get_completed_ids(self) -> Set[str]:
        """Get set of completed question IDs."""
        return self._completed_ids.copy()
    
    def get_all_results(self) -> List[TestResult]:
        """Get all results from checkpoint."""
        return self._results.copy()
    
    def save_question(
        self,
        question_id: str,
        baseline_result: TestResult,
        scaffolded_result: TestResult
    ) -> None:
        """
        Save results for a completed question.
        
        Appends to JSONL file for atomic writes.
        """
        checkpoint = QuestionCheckpoint(
            question_id=question_id,
            baseline_result=baseline_result,
            scaffolded_result=scaffolded_result
        )
        
        # Convert to dict for JSON serialization
        data = {
            'question_id': checkpoint.question_id,
            'baseline_result': asdict(checkpoint.baseline_result),
            'scaffolded_result': asdict(checkpoint.scaffolded_result)
        }
        
        # Append to file
        with open(self.checkpoint_path, 'a') as f:
            f.write(json.dumps(data) + '\n')
        
        # Update in-memory state
        self._completed_ids.add(question_id)
        self._results.append(baseline_result)
        self._results.append(scaffolded_result)
        
        logger.debug(f"Saved checkpoint for question {question_id}")
    
    def get_cumulative_cost(self) -> float:
        """Calculate total cost from all results."""
        return sum(r.cost_usd for r in self._results)
    
    def clear(self) -> None:
        """Clear checkpoint file and in-memory state."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        self._completed_ids = set()
        self._results = []
        logger.info("Checkpoint cleared")


def create_test_result(
    question_id: str,
    subject: str,
    condition: str,
    correct_answer: str,
    model_answer: str,
    input_tokens: int,
    output_tokens: int,
    latency_sec: float,
    cost_usd: float
) -> TestResult:
    """
    Factory function to create a TestResult.
    
    Automatically calculates is_correct based on answers.
    """
    # Normalize answers for comparison
    correct_norm = correct_answer.upper().strip() if correct_answer else ""
    model_norm = model_answer.upper().strip() if model_answer else ""
    
    is_correct = correct_norm == model_norm and model_norm != ""
    
    return TestResult(
        question_id=question_id,
        subject=subject,
        condition=condition,
        correct_answer=correct_answer,
        model_answer=model_answer or "",
        is_correct=is_correct,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_sec=latency_sec,
        cost_usd=cost_usd
    )
