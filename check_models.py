import os
import anthropic

api_key = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

candidates = [
    "claude-sonnet-4-5",          # From search result
    "claude-sonnet-4-5-20250928", # From release date
    "claude-sonnet-4-5-20251001", # Previous guess
    "claude-haiku-4-5-20251001"   # Known working baseline
]

print("Testing model IDs...")

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
