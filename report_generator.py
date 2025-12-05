"""
HTML Report Generator for MMLU-Pro A/B Test Results.

Generates a standalone HTML artifact with visual comparison of baseline vs scaffolded results.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def generate_html_report(
    baseline_accuracy: float,
    baseline_correct: int,
    baseline_total: int,
    baseline_cost: float,
    baseline_cost_per_correct: float,
    scaffolded_accuracy: float,
    scaffolded_correct: int,
    scaffolded_total: int,
    scaffolded_cost: float,
    scaffolded_cost_per_correct: float,
    scaffolding_prompt: str,
    output_path: Path
) -> None:
    """Generate an HTML report comparing baseline vs scaffolded results."""
    
    accuracy_delta = scaffolded_accuracy - baseline_accuracy
    accuracy_delta_sign = "+" if accuracy_delta >= 0 else ""
    accuracy_delta_class = "positive" if accuracy_delta >= 0 else "negative"
    
    cost_delta = scaffolded_cost - baseline_cost
    cost_delta_sign = "+" if cost_delta >= 0 else ""
    
    # Format cost per correct
    baseline_cpc = f"${baseline_cost_per_correct:.4f}" if baseline_cost_per_correct != float('inf') else "N/A"
    scaffolded_cpc = f"${scaffolded_cost_per_correct:.4f}" if scaffolded_cost_per_correct != float('inf') else "N/A"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MMLU-Pro A/B Test Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 40px 20px;
            color: #e0e0e0;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            font-size: 2.2rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .timestamp {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 0.9rem;
        }}
        
        .cards {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .card.baseline {{
            border-left: 4px solid #3498db;
        }}
        
        .card.scaffolded {{
            border-left: 4px solid #9b59b6;
        }}
        
        .card-title {{
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 15px;
        }}
        
        .accuracy {{
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        
        .baseline .accuracy {{
            color: #3498db;
        }}
        
        .scaffolded .accuracy {{
            color: #9b59b6;
        }}
        
        .correct-count {{
            color: #aaa;
            margin-bottom: 20px;
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .metric-label {{
            color: #888;
        }}
        
        .metric-value {{
            font-weight: 600;
        }}
        
        .bar-container {{
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            margin: 10px 0 20px;
            overflow: hidden;
        }}
        
        .bar {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        
        .baseline .bar {{
            background: linear-gradient(90deg, #3498db, #2980b9);
        }}
        
        .scaffolded .bar {{
            background: linear-gradient(90deg, #9b59b6, #8e44ad);
        }}
        
        .delta-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 30px;
        }}
        
        .delta-title {{
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 15px;
        }}
        
        .delta-value {{
            font-size: 2.5rem;
            font-weight: 700;
        }}
        
        .delta-value.positive {{
            color: #2ecc71;
        }}
        
        .delta-value.negative {{
            color: #e74c3c;
        }}
        
        .delta-label {{
            color: #888;
            margin-top: 5px;
        }}
        
        .prompt-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .prompt-title {{
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 15px;
        }}
        
        .prompt-text {{
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9rem;
            color: #a0a0a0;
            line-height: 1.5;
        }}
        
        @media (max-width: 600px) {{
            .cards {{
                grid-template-columns: 1fr;
            }}
            
            .accuracy {{
                font-size: 2.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MMLU-Pro A/B Test Results</h1>
        <p class="timestamp">Generated: {timestamp}</p>
        
        <div class="cards">
            <div class="card baseline">
                <div class="card-title">Baseline (No Scaffolding)</div>
                <div class="accuracy">{baseline_accuracy:.1f}%</div>
                <div class="correct-count">{baseline_correct}/{baseline_total} correct</div>
                <div class="bar-container">
                    <div class="bar" style="width: {baseline_accuracy}%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Cost</span>
                    <span class="metric-value">${baseline_cost:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Cost per Correct</span>
                    <span class="metric-value">{baseline_cpc}</span>
                </div>
            </div>
            
            <div class="card scaffolded">
                <div class="card-title">Scaffolded</div>
                <div class="accuracy">{scaffolded_accuracy:.1f}%</div>
                <div class="correct-count">{scaffolded_correct}/{scaffolded_total} correct</div>
                <div class="bar-container">
                    <div class="bar" style="width: {scaffolded_accuracy}%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Cost</span>
                    <span class="metric-value">${scaffolded_cost:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Cost per Correct</span>
                    <span class="metric-value">{scaffolded_cpc}</span>
                </div>
            </div>
        </div>
        
        <div class="delta-card">
            <div class="delta-title">Accuracy Delta (Scaffolded vs Baseline)</div>
            <div class="delta-value {accuracy_delta_class}">{accuracy_delta_sign}{accuracy_delta:.1f}%</div>
            <div class="delta-label">{"Scaffolding improved accuracy" if accuracy_delta > 0 else "Scaffolding reduced accuracy" if accuracy_delta < 0 else "No change in accuracy"}</div>
        </div>
        
        <div class="prompt-card">
            <div class="prompt-title">Scaffolding Prompt Used</div>
            <div class="prompt-text">{scaffolding_prompt or "(No scaffolding prompt configured)"}</div>
        </div>
    </div>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html)
