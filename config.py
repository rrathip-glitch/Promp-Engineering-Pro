"""
Configuration module for MMLU-Pro A/B Testing Harness.

Loads environment variables, scaffolding config, and defines constants.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# Model Pricing (Input/Output per million tokens)
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-opus-4-5": (15.0, 75.0)
}

# Default model
MODEL_NAME = "claude-haiku-4-5-20251001"
BUDGET_CEILING_USD = 25.0
BUDGET_WARNING_THRESHOLD = 0.8  # Warn at 80% of budget

# Sampling settings
DEFAULT_SAMPLE_SIZE = 100
RANDOM_SEED = 42

# Rate limiting
API_DELAY_SECONDS = 1.0
MAX_RETRIES = 3


@dataclass
class ScaffoldingConfig:
    """Configuration for the scaffolding pre-prompt."""
    enabled: bool
    pre_prompt: str


@dataclass
class Config:
    """Main configuration object."""
    api_key: str
    scaffolding: ScaffoldingConfig
    sample_size: int = DEFAULT_SAMPLE_SIZE
    dry_run: bool = False


def load_api_key() -> str:
    """Load the Anthropic API key from environment variable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it before running the tests."
        )
    return api_key


def load_scaffolding_config(config_path: Optional[Path] = None) -> ScaffoldingConfig:
    """Load scaffolding configuration from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent / "scaffolding_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Scaffolding config file not found: {config_path}. "
            "Please create it with 'enabled' and 'pre_prompt' fields."
        )
    
    with open(config_path, "r") as f:
        data = json.load(f)
    
    return ScaffoldingConfig(
        enabled=data.get("enabled", True),
        pre_prompt=data.get("pre_prompt", "")
    )


def load_config(sample_size: int = DEFAULT_SAMPLE_SIZE, dry_run: bool = False) -> Config:
    """Load complete configuration."""
    return Config(
        api_key=load_api_key(),
        scaffolding=load_scaffolding_config(),
        sample_size=sample_size,
        dry_run=dry_run
    )


def calculate_cost(input_tokens: int, output_tokens: int, model_name: str = MODEL_NAME) -> float:
    """Calculate cost in USD for given token counts and model."""
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING[MODEL_NAME])
    input_price, output_price = pricing
    
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return input_cost + output_cost
