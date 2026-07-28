#!/usr/bin/env python3
"""
Aggregate OpinionQA human responses to calculate Average_Human_Response per question.

This script reads:
- dataset/OpinionQA/questions.json - Question metadata
- dataset/OpinionQA/responses.json - Individual human responses

And outputs:
- dataset/OpinionQA/aggreated/final_question_sheet.json - Aggregated format matching Twin-2k-500-Aggreated
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def aggregate_human_responses(responses: List[Dict]) -> Dict[str, float]:
    """
    Calculate average human response for each question.

    Args:
        responses: List of response dicts with 'Variable_Name' and 'Human_Response'

    Returns:
        Dict mapping Variable_Name to Average_Human_Response
    """
    sums = defaultdict(float)
    counts = defaultdict(int)

    for resp in responses:
        var_name = resp['Variable_Name']
        human_resp = resp['Human_Response']

        # Skip null/None responses
        if human_resp is None:
            continue

        sums[var_name] += human_resp
        counts[var_name] += 1

    # Calculate averages
    averages = {}
    for var_name in sums:
        if counts[var_name] > 0:
            averages[var_name] = sums[var_name] / counts[var_name]

    return averages


def create_aggregated_dataset(
    questions: List[Dict],
    avg_human_responses: Dict[str, float]
) -> Dict[str, List[Dict]]:
    """
    Create aggregated dataset in the same format as Twin-2k-500-Aggreated.

    Args:
        questions: List of question dicts with metadata
        avg_human_responses: Dict of Variable_Name -> Average_Human_Response

    Returns:
        Dict with 'Demographics' and 'Questions' keys
    """
    # For now, put all questions in 'Questions' category
    # TODO: Separate demographics if needed
    aggregated_questions = []

    for q in questions:
        var_name = q['Variable_Name']

        # Skip questions without average response
        if var_name not in avg_human_responses:
            print(f"Warning: No human responses for {var_name}")
            continue

        aggregated_question = {
            'Variable_Name': var_name,
            'Question': q['Question'],
            'Average_LLM_Response': None,  # Will be filled later
            'Average_Human_Response': round(avg_human_responses[var_name], 3),
            'score_range': q['score_range'],
            'Question_Embedding': []  # Empty for now
        }

        aggregated_questions.append(aggregated_question)

    return {
        'Demographics': [],  # Empty for OpinionQA (no demographic questions)
        'Questions': aggregated_questions
    }


def main():
    # Paths
    base_dir = Path('/Users/lijiang/Github/debias_response/dataset/OpinionQA')
    questions_file = base_dir / 'questions.json'
    responses_file = base_dir / 'responses.json'
    output_dir = base_dir / 'aggreated'
    output_file = output_dir / 'final_question_sheet.json'

    # Load data
    print(f"Loading questions from {questions_file}...")
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    print(f"  Loaded {len(questions)} questions")

    print(f"Loading responses from {responses_file}...")
    with open(responses_file, 'r') as f:
        responses = json.load(f)
    print(f"  Loaded {len(responses)} responses")

    # Aggregate
    print("Aggregating human responses...")
    avg_human_responses = aggregate_human_responses(responses)
    print(f"  Calculated averages for {len(avg_human_responses)} questions")

    # Create dataset
    print("Creating aggregated dataset...")
    dataset = create_aggregated_dataset(questions, avg_human_responses)

    # Save
    output_dir.mkdir(exist_ok=True)
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Done! Saved {len(dataset['Questions'])} questions")
    print(f"  - Demographics: {len(dataset['Demographics'])}")
    print(f"  - Questions: {len(dataset['Questions'])}")


if __name__ == '__main__':
    main()
