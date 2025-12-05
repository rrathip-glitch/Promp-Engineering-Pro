import os
import random
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import load_config, ScaffoldingConfig, MODEL_PRICING
from data_loader import load_mmlu_pro_data
from api_client import APIClient, extract_answer_with_llm_fallback
from evaluator import run_single_test

from contextlib import asynccontextmanager
from pathlib import Path

# Global variable for questions
ALL_QUESTIONS = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load data on startup (Synchronous/Blocking for reliability)
    global ALL_QUESTIONS
    try:
        # Use relative path for deployment compatibility
        DATA_DIR = Path(__file__).parent / "MMLU-Pro" / "data"
        logger.info(f"Loading data from {DATA_DIR}")
        
        # Load directly (blocking) to ensure data is ready before serving
        ALL_QUESTIONS = load_mmlu_pro_data(DATA_DIR)
        logger.info(f"Loaded {len(ALL_QUESTIONS)} questions")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        ALL_QUESTIONS = []
    
    yield
    
    # Clean up on shutdown
    ALL_QUESTIONS.clear()

app = FastAPI(title="Prompt Engineering Pro", lifespan=lifespan)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
    )

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway."""
    return {"status": "ok", "questions_loaded": len(ALL_QUESTIONS)}

class TestRequest(BaseModel):
    pre_prompt: str
    model: str
    benchmark: str = "mmlu-pro"

class TestResult(BaseModel):
    status: str
    baseline: Dict[str, Any]
    scaffolded: Dict[str, Any]
    model_used: str
    questions_tested: int

def get_stratified_sample(questions: List[Any], size: int = 14) -> List[Any]:
    """Get a stratified sample of questions (1 per subject if possible)."""
    if not questions:
        return []
        
    subjects = {}
    for q in questions:
        # Handle both dict and Question object
        if isinstance(q, dict):
            subj = q.get('category', 'unknown')
        else:
            subj = q.subject
            
        if subj not in subjects:
            subjects[subj] = []
        subjects[subj].append(q)
    
    sample = []
    # Take one from each subject first
    for subj in sorted(subjects.keys()):
        if subjects[subj]:
            sample.append(random.choice(subjects[subj]))
    
    # Fill remaining if needed (or trim if too many subjects)
    if len(sample) > size:
        sample = random.sample(sample, size)
    elif len(sample) < size:
        remaining_needed = size - len(sample)
        # Flatten all remaining questions
        all_remaining = []
        
        # Get IDs for exclusion
        sample_ids = set()
        for q in sample:
            if isinstance(q, dict):
                sample_ids.add(q.get('question_id'))
            else:
                sample_ids.add(q.question_id)
                
        for q in questions:
            q_id = q.get('question_id') if isinstance(q, dict) else q.question_id
            if q_id not in sample_ids:
                all_remaining.append(q)
        
        if all_remaining:
            sample.extend(random.sample(all_remaining, min(len(all_remaining), remaining_needed)))
            
    return sample

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

from fastapi.responses import StreamingResponse
import json
import asyncio

@app.post("/api/run-test")
async def run_test(request: TestRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
    
    client = APIClient(api_key=api_key)
    
    # Get stratified sample
    questions = get_stratified_sample(ALL_QUESTIONS, size=14)
    total_steps = len(questions) * 2  # Baseline + Scaffolded
    
    async def event_generator():
        # Run Baseline (No scaffolding)
        baseline_correct = 0
        baseline_cost = 0.0
        baseline_config = ScaffoldingConfig(enabled=False, pre_prompt="")
        
        completed_steps = 0
        
        for i, q in enumerate(questions):
            # Yield progress
            yield json.dumps({
                "type": "progress",
                "completed": completed_steps,
                "total": total_steps,
                "message": f"Running Baseline: Question {i+1}/{len(questions)}"
            }) + "\n"
            
            # Small delay to ensure UI updates smoothly
            await asyncio.sleep(0.01)
            
            answer, response = run_single_test(client, q, baseline_config, include_scaffolding=False, model=request.model)
            baseline_cost += response.cost_usd
            
            correct_answer = q.correct_answer if not isinstance(q, dict) else q.get('answer')
            if answer == correct_answer:
                baseline_correct += 1
            
            completed_steps += 1
            
        # Run Scaffolded
        scaffolded_correct = 0
        scaffolded_cost = 0.0
        
        # Optimization: If pre-prompt is empty, reuse baseline results
        is_empty_scaffolding = not request.pre_prompt or not request.pre_prompt.strip()
        
        if is_empty_scaffolding:
            logger.info("Empty scaffolding detected. Reusing baseline results.")
            scaffolded_correct = baseline_correct
            scaffolded_cost = baseline_cost
            
            # Simulate progress for the second half
            for i in range(len(questions)):
                completed_steps += 1
                yield json.dumps({
                    "type": "progress",
                    "completed": completed_steps,
                    "total": total_steps,
                    "message": f"Skipping redundant test {i+1}/{len(questions)}"
                }) + "\n"
                await asyncio.sleep(0.05) # Fast forward
                
        else:
            scaffolded_config = ScaffoldingConfig(enabled=True, pre_prompt=request.pre_prompt)
            
            for i, q in enumerate(questions):
                # Yield progress
                yield json.dumps({
                    "type": "progress",
                    "completed": completed_steps,
                    "total": total_steps,
                    "message": f"Running Scaffolded: Question {i+1}/{len(questions)}"
                }) + "\n"
                
                await asyncio.sleep(0.01)
                
                answer, response = run_single_test(client, q, scaffolded_config, include_scaffolding=True, model=request.model)
                scaffolded_cost += response.cost_usd
                
                correct_answer = q.correct_answer if not isinstance(q, dict) else q.get('answer')
                if answer == correct_answer:
                    scaffolded_correct += 1
                
                completed_steps += 1

        # Final Result
        result_data = {
            "status": "complete",
            "baseline": {
                "total_questions": len(questions),
                "correct": baseline_correct,
                "accuracy_pct": round((baseline_correct / len(questions)) * 100, 1),
                "total_cost_usd": round(baseline_cost, 4),
                "cost_per_correct_usd": round(baseline_cost / baseline_correct, 5) if baseline_correct > 0 else 0
            },
            "scaffolded": {
                "total_questions": len(questions),
                "correct": scaffolded_correct,
                "accuracy_pct": round((scaffolded_correct / len(questions)) * 100, 1),
                "total_cost_usd": round(scaffolded_cost, 4),
                "cost_per_correct_usd": round(scaffolded_cost / scaffolded_correct, 5) if scaffolded_correct > 0 else 0
            },
            "model_used": request.model,
            "questions_tested": len(questions)
        }
        
        yield json.dumps({
            "type": "result",
            "data": result_data
        }) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
