"""
Data loader module for MMLU-Pro dataset.

Handles loading, parsing, and stratified sampling of questions.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from config import RANDOM_SEED, DEFAULT_SAMPLE_SIZE


logger = logging.getLogger(__name__)


@dataclass
class Question:
    """Represents a single MMLU-Pro question."""
    question_id: str
    subject: str
    question_text: str
    options: List[str]
    correct_answer: str  # Letter A-J
    
    def format_options(self) -> str:
        """Format options as lettered choices."""
        letters = "ABCDEFGHIJ"
        formatted = []
        for i, option in enumerate(self.options):
            if i < len(letters):
                formatted.append(f"{letters[i]}. {option}")
        return "\n".join(formatted)


def load_parquet_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from a parquet file."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required to load parquet files. Install with: pip install pandas pyarrow")
    
    df = pd.read_parquet(file_path)
    return df.to_dict('records')


def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        # Some formats have data nested under a key
        if 'data' in data:
            return data['data']
        elif 'questions' in data:
            return data['questions']
        else:
            # Convert dict of dicts to list
            return list(data.values())
    return data


def load_jsonl_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_question(raw: Dict[str, Any], index: int) -> Optional[Question]:
    """Parse a raw question dict into a Question object."""
    try:
        # Handle different field naming conventions in MMLU-Pro
        question_id = str(raw.get('question_id', raw.get('id', index)))
        subject = raw.get('category', raw.get('subject', raw.get('topic', 'unknown')))
        question_text = raw.get('question', raw.get('question_text', ''))
        
        # Get options - could be list, ndarray, or dict
        options = raw.get('options', [])
        # Convert numpy array to list if needed
        if hasattr(options, 'tolist'):
            options = options.tolist()
        elif isinstance(options, dict):
            options = list(options.values())
        
        # Get correct answer - prefer 'answer' field first (letter), then 'answer_index' (int)
        answer = raw.get('answer', '')
        answer_index = raw.get('answer_index', None)
        
        # If answer is a string letter, use it directly  
        if isinstance(answer, str) and answer.strip() and answer.strip()[0] in 'ABCDEFGHIJ':
            answer = answer.strip()[0].upper()
        # Otherwise convert index to letter
        elif answer_index is not None:
            # Handle numpy integers
            if hasattr(answer_index, 'item'):
                answer_index = answer_index.item()
            if isinstance(answer_index, (int, float)):
                answer = chr(ord('A') + int(answer_index))
            elif isinstance(answer_index, str) and answer_index.isdigit():
                answer = chr(ord('A') + int(answer_index))
        elif isinstance(answer, (int, float)):
            answer = chr(ord('A') + int(answer))
        elif isinstance(answer, str) and answer.isdigit():
            answer = chr(ord('A') + int(answer))
        else:
            answer = str(answer).upper().strip()
            if answer and answer[0] in 'ABCDEFGHIJ':
                answer = answer[0]
        
        # Validate we have required fields
        if not question_text:
            logger.debug(f"Skipping question at index {index}: missing question_text")
            return None
        if not options or len(options) == 0:
            logger.debug(f"Skipping question at index {index}: missing options")
            return None
        if not answer or answer not in 'ABCDEFGHIJ':
            logger.debug(f"Skipping question at index {index}: invalid answer '{answer}'")
            return None
            
        return Question(
            question_id=question_id,
            subject=subject,
            question_text=question_text,
            options=list(options),  # Ensure it's a plain list
            correct_answer=answer
        )
    except Exception as e:
        logger.warning(f"Error parsing question at index {index}: {e}")
        return None


def load_mmlu_pro_data(data_dir: Union[Path, str]) -> List[Question]:
    """
    Load MMLU-Pro dataset from a directory.
    
    Supports parquet, JSON, and JSONL formats.
    Will search for common file patterns.
    """
    questions = []
    
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    
    # Search for data files
    file_patterns = [
        "*.parquet",
        "test*.parquet",
        "validation*.parquet", 
        "*.json",
        "*.jsonl"
    ]
    
    # Use a set to avoid duplicates from overlapping patterns
    data_files = set()
    for pattern in file_patterns:
        data_files.update(data_dir.glob(pattern))
    
    # Also check subdirectories if data_dir exists
    if data_dir.exists():
        for subdir in data_dir.iterdir():
            if subdir.is_dir():
                for pattern in file_patterns:
                    data_files.update(subdir.glob(pattern))
    
    # Convert back to sorted list for consistent order
    data_files = sorted(list(data_files))
    
    if not data_files:
        raise FileNotFoundError(
            f"No data files found in {data_dir}. "
            "Please ensure the MMLU-Pro dataset is downloaded."
        )
    
    logger.info(f"Found {len(data_files)} data files")
    
    # Load all data
    raw_data = []
    for file_path in data_files:
        logger.info(f"Loading {file_path}")
        try:
            if file_path.suffix == '.parquet':
                raw_data.extend(load_parquet_data(file_path))
            elif file_path.suffix == '.jsonl':
                raw_data.extend(load_jsonl_data(file_path))
            elif file_path.suffix == '.json':
                raw_data.extend(load_json_data(file_path))
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
    
    logger.info(f"Loaded {len(raw_data)} raw questions")
    
    # Parse questions
    for i, raw in enumerate(raw_data):
        q = parse_question(raw, i)
        if q:
            questions.append(q)
    
    logger.info(f"Parsed {len(questions)} valid questions")
    return questions


def stratified_sample(
    questions: List[Question],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = RANDOM_SEED
) -> List[Question]:
    """
    Perform stratified random sampling across subjects.
    
    Ensures even distribution of questions across different subjects.
    """
    random.seed(seed)
    
    # Group by subject
    by_subject: Dict[str, List[Question]] = {}
    for q in questions:
        if q.subject not in by_subject:
            by_subject[q.subject] = []
        by_subject[q.subject].append(q)
    
    num_subjects = len(by_subject)
    if num_subjects == 0:
        return []
    
    # Calculate questions per subject
    per_subject = max(1, sample_size // num_subjects)
    remainder = sample_size - (per_subject * num_subjects)
    
    logger.info(f"Sampling {per_subject} questions from each of {num_subjects} subjects")
    
    sampled = []
    subjects = sorted(by_subject.keys())  # Sort for reproducibility
    
    for i, subject in enumerate(subjects):
        subject_questions = by_subject[subject]
        # Add one extra for first 'remainder' subjects to reach exact sample_size
        n_to_sample = per_subject + (1 if i < remainder else 0)
        n_to_sample = min(n_to_sample, len(subject_questions))
        
        sample = random.sample(subject_questions, n_to_sample)
        sampled.extend(sample)
        logger.debug(f"Sampled {len(sample)} from {subject}")
    
    # Shuffle final sample for randomized processing order
    random.shuffle(sampled)
    
    logger.info(f"Final sample size: {len(sampled)} questions from {num_subjects} subjects")
    return sampled


def load_and_sample(
    data_dir: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = RANDOM_SEED
) -> List[Question]:
    """
    Load MMLU-Pro data and return a stratified sample.
    
    This is the main entry point for data loading.
    """
    all_questions = load_mmlu_pro_data(data_dir)
    
    if len(all_questions) <= sample_size:
        logger.info(f"Dataset size ({len(all_questions)}) <= sample size, using all questions")
        return all_questions
    
    return stratified_sample(all_questions, sample_size, seed)
