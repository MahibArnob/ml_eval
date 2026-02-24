
import os
import json
import pandas as pd
from django.conf import settings
import sys

# Add project root to sys.path to allow importing apps
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup Django environment
import django_setup
django_setup.setup_django()

# Now we can import from assessments
from assessments.codeToMQL import convert_code_to_mql_pipeline, grade_mql
from assessments.services.llm_service import LLMService

ASSIGNMENTS_FILE = 'ml_eval_framework/datasets/ml_assignments_100.json'
RESULTS_FILE = 'ml_eval_framework/results/evaluation_results_comparison.jsonl'

def load_assignments():
    if not os.path.exists(ASSIGNMENTS_FILE):
        return []
    with open(ASSIGNMENTS_FILE, 'r') as f:
        return json.load(f)

def grade_with_chatgpt(student_code, question_text, reference_code):
    """Grade student code using raw ChatGPT API."""
    prompt = f"""
    You are an AI Grader for a Machine Learning course.
    
    Question: {question_text}
    
    Reference Solution:
    {reference_code}
    
    Student Solution:
    {student_code}
    
    Task:
    1. Evaluate the Student Solution for correctness.
    2. Provide a score out of 10.0.
    3. Provide brief feedback.
    
    Return JSON:
    {{
        "score": <float>,
        "feedback": "<string>"
    }}
    """
    
    structured_format = {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "feedback": {"type": "string"}
        },
        "required": ["score", "feedback"],
        "additionalProperties": False
    }
    
    try:
        response = LLMService.ask(
            prompt=prompt,
            structured_output=structured_format,
            provider="openai"
        )
        return response
    except Exception as e:
        print(f"Error grading with ChatGPT: {e}")
        return {"score": 0.0, "feedback": f"Error: {e}"}

def evaluate_mql(student_code, assignment):
    """Run MQL-based evaluation."""
    dataset_path = assignment['dataset_path']
    mql_result = convert_code_to_mql_pipeline(student_code, dataset_path)
    
    if not mql_result['success']:
        return {
            "score": 0.0,
            "feedback": f"MQL Conversion Failed: {mql_result.get('errors')}",
            "semantic_match": False
        }
        
    student_mql = mql_result['mql_queries']
    instructor_mql = assignment['reference_mql']
    
    grading_result = grade_mql(
        instructor_mql=instructor_mql,
        student_mql=student_mql,
        student_python_code=student_code,
        question_text=assignment['question_text'],
        dataset_context=assignment['dataset_context'],
        max_points=10.0
    )
    return {
        "score": grading_result.get('score_awarded', 0.0),
        "feedback": grading_result.get('detailed_feedback', ''),
        "semantic_match": grading_result.get('semantic_match', False)
    }

def main():
    assignments = load_assignments()
    if not assignments:
        print(f"No assignments found in {ASSIGNMENTS_FILE}. Run generate_ml_dataset.py first.")
        return

    print(f"Loaded {len(assignments)} assignments.")
    
    results = []

    for assignment in assignments:
        print(f"\nEvaluated Assignment: {assignment['id']}")
        
        # 1. Correct Solution Evaluation
        correct_code = assignment['reference_code']
        
        # MQL
        print("  Eval: Correct Code (MQL)")
        mql_correct = evaluate_mql(correct_code, assignment)
        
        # GPT
        print("  Eval: Correct Code (GPT)")
        gpt_correct = grade_with_chatgpt(correct_code, assignment['question_text'], assignment['reference_code'])
        
        # 2. Buggy Solution Evaluation
        buggy_code = assignment.get('buggy_code', '')
        if not buggy_code:
            print("  Skipping Buggy Code Eval (Not generated)")
            mql_buggy = {"score": 0.0, "feedback": "No buggy code"}
            gpt_buggy = {"score": 0.0, "feedback": "No buggy code"}
        else:
            # MQL
            print("  Eval: Buggy Code (MQL)")
            mql_buggy = evaluate_mql(buggy_code, assignment)
            
            # GPT
            print("  Eval: Buggy Code (GPT)")
            gpt_buggy = grade_with_chatgpt(buggy_code, assignment['question_text'], assignment['reference_code'])

        result_entry = {
            "assignment_id": assignment['id'],
            "mql_correct": mql_correct,
            "gpt_correct": gpt_correct,
            "mql_buggy": mql_buggy,
            "gpt_buggy": gpt_buggy
        }
        results.append(result_entry)

    # Save results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
            
    print(f"\nEvaluation complete. Results saved to {RESULTS_FILE}")

if __name__ == '__main__':
    main()
