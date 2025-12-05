# MMLU-Pro Test Setup Instructions

## ✅ Completed Steps

1. ✅ **Dependencies Installed** - All Python packages are installed
2. ✅ **Scaffolding Configured** - Pre-prompt is set up with step-by-step reasoning guidance

## 🔴 Required Steps (Manual)

### 1. Download the MMLU-Pro Dataset

The dataset needs to be downloaded from HuggingFace. Run these commands:

```bash
cd /Users/rathiprajakumar/.gemini/antigravity/scratch/mmlu_pro_tester

# Install huggingface-hub if not already installed
pip3 install huggingface-hub

# Download the dataset
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='TIGER-Lab/MMLU-Pro',
    repo_type='dataset',
    local_dir='./MMLU-Pro/data',
    ignore_patterns=['*.md', '*.git*']
)
"
```

**Alternative method** (if the above doesn't work):
```bash
# Use the datasets library
pip3 install datasets
python3 -c "
from datasets import load_dataset
dataset = load_dataset('TIGER-Lab/MMLU-Pro')
# Save to local directory
dataset['validation'].to_parquet('./MMLU-Pro/data/validation.parquet')
dataset['test'].to_parquet('./MMLU-Pro/data/test.parquet')
"
```

### 2. Set Your Anthropic API Key

You need to set your API key as an environment variable. Choose one method:

**Option A: Set for current terminal session**
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-YOUR-API-KEY-HERE"
```

**Option B: Add to your shell profile (permanent)**
```bash
# For zsh (default on macOS)
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-YOUR-API-KEY-HERE"' >> ~/.zshrc
source ~/.zshrc

# For bash
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-YOUR-API-KEY-HERE"' >> ~/.bashrc
source ~/.bashrc
```

**Option C: Create a .env file (if you prefer)**
```bash
echo 'ANTHROPIC_API_KEY=sk-ant-api03-YOUR-API-KEY-HERE' > .env
```
Then modify `config.py` to load from .env file.

### 3. Get Your Anthropic API Key

1. Go to: https://console.anthropic.com/
2. Sign in or create an account
3. Navigate to "API Keys" section
4. Create a new API key or copy an existing one
5. The key will start with `sk-ant-api03-`

## 🚀 Running the Test

Once you've completed the above steps:

```bash
cd /Users/rathiprajakumar/.gemini/antigravity/scratch/mmlu_pro_tester

# Verify dataset is downloaded
ls -la MMLU-Pro/data/

# Verify API key is set
echo $ANTHROPIC_API_KEY

# Run a small test first (10 questions)
python3 main.py --sample-size 10

# Run the full test (100 questions, default)
python3 main.py

# Or customize further
python3 main.py --sample-size 50 --verbose
```

## 📊 Understanding the Output

The test will create:
- `checkpoint.jsonl` - Progress checkpoint (resume if interrupted)
- `results_detailed.csv` - Per-question results
- `results_summary.csv` - Aggregated statistics
- Terminal output with accuracy comparison

## 💰 Cost Estimate

- Budget ceiling: $25.00
- Estimated cost for 100 questions: ~$2-5 (depends on question length)
- The test will automatically stop if budget is exceeded

## ⚠️ Important Notes

- The test runs both baseline AND scaffolded conditions for each question
- Total API calls = 2 × number of questions
- Progress is saved, you can interrupt and resume safely
- Use `--clear-checkpoint` to start fresh
