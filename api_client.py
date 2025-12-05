"""
Anthropic API client module with cost tracking and budget safety.

Handles all API calls to Claude with retry logic and spending limits.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import re

from config import (
    MODEL_NAME,
    BUDGET_CEILING_USD,
    BUDGET_WARNING_THRESHOLD,
    API_DELAY_SECONDS,
    MAX_RETRIES,
    calculate_cost
)


logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Response from an API call."""
    answer_text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float
    

class BudgetExceededError(Exception):
    """Raised when the budget ceiling would be exceeded."""
    pass


class APIClient:
    """
    Anthropic API client with cost tracking.
    
    Tracks cumulative spend and enforces budget ceiling.
    """
    
    def __init__(self, api_key: str, dry_run: bool = False):
        """
        Initialize the API client.
        
        Args:
            api_key: Anthropic API key
            dry_run: If True, simulate API calls without actually calling
        """
        self.api_key = api_key
        self.dry_run = dry_run
        self.cumulative_cost = 0.0
        self.total_calls = 0
        self.budget_warning_issued = False
        
        if not dry_run:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package is required. Install with: pip install anthropic"
                )
    
    def check_budget(self, estimated_cost: float = 0.01) -> None:
        """
        Check if making another call would exceed budget.
        
        Args:
            estimated_cost: Estimated cost of the next call
            
        Raises:
            BudgetExceededError: If budget would be exceeded
        """
        if self.cumulative_cost + estimated_cost > BUDGET_CEILING_USD:
            raise BudgetExceededError(
                f"Budget ceiling of ${BUDGET_CEILING_USD:.2f} would be exceeded. "
                f"Current spend: ${self.cumulative_cost:.4f}"
            )
        
        # Issue warning at 80% threshold
        if not self.budget_warning_issued:
            if self.cumulative_cost >= BUDGET_CEILING_USD * BUDGET_WARNING_THRESHOLD:
                logger.warning(
                    f"⚠️  Budget warning: 80% of ${BUDGET_CEILING_USD:.2f} budget consumed. "
                    f"Current spend: ${self.cumulative_cost:.4f}"
                )
                self.budget_warning_issued = True
    
    def call(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1024,
        model: str = MODEL_NAME
    ) -> APIResponse:
        """
        Make an API call to Claude.
        
        Args:
            system_prompt: System message for the model
            user_message: User message (question + options)
            max_tokens: Maximum tokens in response
            
        Returns:
            APIResponse with answer, tokens, latency, and cost
        """
        # Check budget before call
        self.check_budget()
        
        if self.dry_run:
            return self._simulate_call(user_message, model)
        
        return self._real_call(system_prompt, user_message, max_tokens, model)
    
    def _simulate_call(self, user_message: str, model: str) -> APIResponse:
        """Simulate an API call for dry run mode."""
        # Simulate some delay
        time.sleep(0.1)
        
        # Estimate tokens (rough approximation)
        input_tokens = len(user_message.split()) * 1.3
        output_tokens = 5
        
        cost = calculate_cost(int(input_tokens), int(output_tokens), model)
        self.cumulative_cost += cost
        self.total_calls += 1
        
        # Return random answer for simulation
        import random
        answer = random.choice("ABCDEFGHIJ")
        
        return APIResponse(
            answer_text=answer,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            latency_seconds=0.1,
            cost_usd=cost
        )
    
    def _real_call(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        model: str
    ) -> APIResponse:
        """Make actual API call with retry logic."""
        import anthropic
        
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message}
                    ]
                )
                
                latency = time.time() - start_time
                
                # Extract text from response
                answer_text = ""
                if response.content and len(response.content) > 0:
                    answer_text = response.content[0].text
                
                # Get token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                
                # Calculate cost
                cost = calculate_cost(input_tokens, output_tokens, model)
                self.cumulative_cost += cost
                self.total_calls += 1
                
                # Delay before next call
                time.sleep(API_DELAY_SECONDS)
                
                return APIResponse(
                    answer_text=answer_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=latency,
                    cost_usd=cost
                )
                
            except anthropic.RateLimitError as e:
                last_error = e
                wait_time = (2 ** attempt) * 2  # Exponential backoff
                logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                
            except anthropic.APIStatusError as e:
                last_error = e
                wait_time = (2 ** attempt) * 1
                logger.warning(f"API error: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                else:
                    raise
        
        raise last_error if last_error else Exception("Max retries exceeded")
    
    def get_stats(self) -> Tuple[float, int]:
        """Return cumulative cost and total calls."""
        return self.cumulative_cost, self.total_calls


def extract_answer(response_text: str) -> Optional[str]:
    """
    Extract the answer letter (A-J) from model response.
    
    Handles various response formats:
    - Single letter: "A"
    - With punctuation: "A."
    - With explanation: "A. The answer is..."
    - Full sentence: "The answer is A"
    
    Returns:
        The extracted letter (A-J) or None if no valid answer found
    """
    if not response_text:
        return None
    
    text = response_text.strip()
    text_upper = text.upper()
    
    # Case 1: Response is just a single letter (possibly with punctuation)
    clean = text_upper.strip('.')
    if len(clean) == 1 and clean in 'ABCDEFGHIJ':
        return clean
    
    # Case 2: Response starts with a valid answer letter followed by punctuation or space
    # But not if it starts with common words like "I think", "A good", etc.
    if len(text_upper) >= 1:
        first_char = text_upper[0]
        if first_char in 'ABCDEFGHIJ':
            # Check if it's followed by punctuation or period (indicating answer)
            if len(text_upper) == 1:
                return first_char
            second_char = text_upper[1] if len(text_upper) > 1 else ''
            if second_char in '.):, ' and first_char not in 'AI':
                # A and I could be words, be more careful
                return first_char
            elif second_char in '.):' and first_char in 'AI':
                return first_char
    
    # Case 3: Look for explicit patterns like "answer is X" or "answer: X"
    patterns = [
        r'\b(?:answer|choice|option)\s*(?:is|:)\s*([A-J])\b',
        r'\b([A-J])\s+is\s+(?:the\s+)?(?:correct|right|best)\b',
        r'\b(?:correct|right|best)\s+(?:answer|choice|option)\s*(?:is|:)?\s*([A-J])\b',
        r'^\s*\(?([B-HJ])\)?[\.:\)]',  # Letter at start (excluding A and I which could be words)
        r'^\s*\(([A-J])\)',  # Letter in parentheses at start
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_upper, re.IGNORECASE)
        if match:
            # Find the first non-None group
            for g in match.groups():
                if g:
                    return g.upper()
    
    # Case 4: Look for letter followed by period or colon (common answer format)
    match = re.search(r'\b([B-HJ])[\.:\)]', text_upper)
    if match:
        return match.group(1)
    
    # Case 5: For A, only match if it's clearly an answer (followed by period at word boundary)
    match = re.search(r'(?<![A-Za-z])([A])\.(?!\w)', text_upper)
    if match:
        return match.group(1)
    
    # Case 6: Last resort - find standalone letter that's not A or I (common words)
    match = re.search(r'\b([B-HJ])\b', text_upper)
    if match:
        return match.group(1)
    
    return None


def extract_answer_with_llm_fallback(
    response_text: str,
    client: 'APIClient',
    question_text: str = "",
    options_text: str = ""
) -> str:
    """
    Extract answer using regex first, then LLM fallback if needed.
    
    Guarantees a valid A-J letter is returned.
    
    Args:
        response_text: The model's response to extract from
        client: API client for LLM fallback
        question_text: Original question (for context in fallback)
        options_text: Original options (for context in fallback)
        
    Returns:
        A valid letter A-J (guaranteed)
    """
    # Try regex extraction first
    extracted = extract_answer(response_text)
    if extracted and extracted in 'ABCDEFGHIJ':
        return extracted
    
    # LLM fallback: Ask Haiku to extract the answer
    fallback_system = (
        "You are an answer extraction assistant. Your ONLY job is to output "
        "a single letter A-J representing the answer choice. "
        "Output ONLY the letter, nothing else."
    )
    
    fallback_prompt = f"""The following is a response to a multiple choice question. 
Extract which answer (A-J) was chosen or intended.

Question: {question_text[:300]}...

Options: {options_text[:500]}...

Response to extract from: "{response_text}"

Output ONLY the single letter (A-J) that best represents the answer. If unclear, make your best guess."""

    try:
        fallback_response = client.call(fallback_system, fallback_prompt, max_tokens=5)
        fallback_text = fallback_response.answer_text.strip().upper()
        
        # Extract letter from fallback response
        for char in fallback_text:
            if char in 'ABCDEFGHIJ':
                return char
        
        # If still no valid letter, default to A
        return 'A'
        
    except Exception as e:
        logger.warning(f"LLM fallback extraction failed: {e}")
        # Default to A if all else fails
        return 'A'
