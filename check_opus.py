import os
import anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

candidates = [
    "claude-opus-4-5",
    "claude-opus-4-5-20251001",
    "claude-3-opus-20240229" # Baseline check
]

print("Testing Opus model IDs...")

for model in candidates:
    print(f"Testing {model}...", end=" ")
    try:
        client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print("SUCCESS! ✅")
    except anthropic.NotFoundError:
        print("Not Found ❌")
    except Exception as e:
        print(f"Error: {e}")
