
import json
import os
import pandas as pd
import numpy as np

RESULTS_FILE = 'ml_eval_framework/results/evaluation_results_comparison.jsonl'
OUTPUT_REPORT = 'ml_eval_framework/results/analysis_report_comparison.txt'

def load_results():
    if not os.path.exists(RESULTS_FILE):
        return []
    results = []
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def analyze(results):
    if not results:
        print("No results to analyze.")
        return

    df = pd.DataFrame(results)
    
    # Extract scores
    mql_correct_scores = [r['mql_correct']['score'] for r in results]
    gpt_correct_scores = [r['gpt_correct']['score'] for r in results]
    mql_buggy_scores = [r['mql_buggy']['score'] for r in results]
    gpt_buggy_scores = [r['gpt_buggy']['score'] for r in results]
    
    # Thresholds
    PASS_THRESHOLD = 7.0
    
    # Calculate Metrics
    # 1. Correct Solution (Should Pass)
    mql_tp = sum(s >= PASS_THRESHOLD for s in mql_correct_scores)
    gpt_tp = sum(s >= PASS_THRESHOLD for s in gpt_correct_scores)
    
    # 2. Buggy Solution (Should Fail)
    # Note: If buggy code is detected, score should be low. 
    # "Success" here means correctly failing the student (Score < Threshold)
    mql_tn = sum(s < PASS_THRESHOLD for s in mql_buggy_scores)
    gpt_tn = sum(s < PASS_THRESHOLD for s in gpt_buggy_scores)
    
    total = len(results)
    
    report = f"""
    ML Evaluation Framework Comparison Report
    =========================================
    Total Assignments: {total}
    
    Metric 1: Correct Code Recognition (True Positive Rate)
    -------------------------------------------------------
    MQL System Pass Rate: {mql_tp}/{total} ({mql_tp/total*100:.2f}%)
    ChatGPT Pass Rate:    {gpt_tp}/{total} ({gpt_tp/total*100:.2f}%)
    
    Metric 2: Buggy Code Detection (True Negative Rate)
    ---------------------------------------------------
    MQL System Rejection Rate: {mql_tn}/{total} ({mql_tn/total*100:.2f}%)
    ChatGPT Rejection Rate:    {gpt_tn}/{total} ({gpt_tn/total*100:.2f}%)
    
    Average Scores
    --------------
    MQL Correct: {np.mean(mql_correct_scores):.2f}
    MQL Buggy:   {np.mean(mql_buggy_scores):.2f}
    GPT Correct: {np.mean(gpt_correct_scores):.2f}
    GPT Buggy:   {np.mean(gpt_buggy_scores):.2f}
    
    Detailed Sample (First 3)
    -------------------------
    """
    
    for i in range(min(3, len(results))):
        r = results[i]
        report += f"\nAssignment: {r['assignment_id']}\n"
        report += f"  MQL Correct: {r['mql_correct']['score']} | Feedback: {r['mql_correct']['feedback'][:50]}...\n"
        report += f"  GPT Correct: {r['gpt_correct']['score']} | Feedback: {r['gpt_correct']['feedback'][:50]}...\n"
        report += f"  MQL Buggy:   {r['mql_buggy']['score']}   | Feedback: {r['mql_buggy']['feedback'][:50]}...\n"
        report += f"  GPT Buggy:   {r['gpt_buggy']['score']}   | Feedback: {r['gpt_buggy']['feedback'][:50]}...\n"
        report += "-" * 50
        
    print(report)
    
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)
    print(f"Report saved to {OUTPUT_REPORT}")

if __name__ == '__main__':
    results = load_results()
    analyze(results)
