#!/usr/bin/env python3
"""
Pre-flight check script to verify all setup requirements before running the test.
"""

import os
import sys
from pathlib import Path


def check_api_key():
    """Check if API key is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print("   Get your key at: https://console.anthropic.com/")
        return False
    elif not api_key.startswith("sk-ant-"):
        print("⚠️  API key doesn't look right (should start with 'sk-ant-')")
        return False
    else:
        masked = api_key[:12] + "..." + api_key[-4:]
        print(f"✅ API key found: {masked}")
        return True


def check_dependencies():
    """Check if required Python packages are installed."""
    required = {
        'anthropic': '0.40.0',
        'pandas': '2.0.0',
        'tqdm': '4.65.0',
        'pyarrow': '14.0.0'
    }
    
    all_ok = True
    for package, min_version in required.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {package} ({version})")
        except ImportError:
            print(f"❌ {package} not installed")
            all_ok = False
    
    return all_ok


def check_dataset():
    """Check if MMLU-Pro dataset is downloaded."""
    data_dir = Path(__file__).parent / "MMLU-Pro" / "data"
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run: python3 download_dataset.py")
        return False
    
    # Check for data files
    parquet_files = list(data_dir.glob("*.parquet"))
    json_files = list(data_dir.glob("*.json*"))
    
    if not parquet_files and not json_files:
        print(f"❌ No data files found in {data_dir}")
        print("   Run: python3 download_dataset.py")
        return False
    
    total_files = len(parquet_files) + len(json_files)
    print(f"✅ Dataset found ({total_files} files in {data_dir})")
    
    for f in parquet_files[:3]:  # Show first 3
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name} ({size_mb:.1f} MB)")
    
    return True


def check_config():
    """Check if scaffolding config is set up."""
    config_path = Path(__file__).parent / "scaffolding_config.json"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    import json
    with open(config_path) as f:
        config = json.load(f)
    
    if config.get("pre_prompt") == "YOUR_SCAFFOLDING_TEXT_HERE":
        print("⚠️  Scaffolding config has placeholder text")
        print("   (Still usable, but consider customizing the pre-prompt)")
        return True
    
    preview = config.get("pre_prompt", "")[:60]
    print(f"✅ Scaffolding config: \"{preview}...\"")
    return True


def main():
    print("=" * 70)
    print("MMLU-Pro Test - Pre-flight Check")
    print("=" * 70)
    print()
    
    checks = {
        "Python Dependencies": check_dependencies(),
        "Scaffolding Config": check_config(),
        "Dataset Files": check_dataset(),
        "API Key": check_api_key(),
    }
    
    print()
    print("=" * 70)
    
    if all(checks.values()):
        print("✅ ALL CHECKS PASSED!")
        print()
        print("You're ready to run the test:")
        print("  python3 main.py --sample-size 10    # Start with 10 questions")
        print("  python3 main.py                     # Run full 100 questions")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print()
        print("Failed checks:")
        for name, passed in checks.items():
            if not passed:
                print(f"  - {name}")
        print()
        print("Please fix the issues above before running the test.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
