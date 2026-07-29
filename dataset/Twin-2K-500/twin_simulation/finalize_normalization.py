import pandas as pd
import json
import os

def normalize(value, score_range):
    """Normalizes a score to a 0-1 scale based on the provided score_range."""
    min_score, max_score = score_range
    if (max_score - min_score) == 0:
        return 0
    return (value - min_score) / (max_score - min_score)

def process_and_normalize_final():
    """
    Calculates LLM averages, fills them in the original JSON structure,
    and adds normalized versions of both human and LLM scores.
    """
    print("--- Starting Final Analysis and Normalization ---")

    # --- Path Setup ---
    try:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    except NameError:
        PROJECT_ROOT = os.path.abspath('.')
    model = 'DeepSeek-V3'
    SIMULATION_CSV_PATH = os.path.join(PROJECT_ROOT, 'dataset', model, 'simulated_responses_consolidated.csv')
    SURVEY_JSON_PATH = os.path.join(PROJECT_ROOT, 'dataset', model, 'likert_question_final.json')
    OUTPUT_JSON_PATH = os.path.join(PROJECT_ROOT, 'dataset', model, 'likert_question_final.json') # New output file

    # --- File Existence Check ---
    if not os.path.exists(SIMULATION_CSV_PATH):
        print(f"Error: Simulation data file not found at '{SIMULATION_CSV_PATH}'")
        return
    if not os.path.exists(SURVEY_JSON_PATH):
        print(f"Error: Original survey JSON file not found at '{SURVEY_JSON_PATH}'")
        return

    # --- Load Data ---
    print(f"Loading simulated responses from {SIMULATION_CSV_PATH}...")
    try:
        df = pd.read_csv(SIMULATION_CSV_PATH)
    except pd.errors.EmptyDataError:
        print(f"Error: The simulation data file is empty.")
        return

    print(f"Loading survey structure from {SURVEY_JSON_PATH}...")
    with open(SURVEY_JSON_PATH, 'r', encoding='utf-8') as f:
        survey_data = json.load(f)

    # --- Analysis and Transformation ---
    print("Processing scores, filling LLM responses, and adding normalized values...")

    final_data = {}

    for category, questions in survey_data.items():
        if not isinstance(questions, list):
            final_data[category] = questions # Preserve non-list items if any
            continue

        new_question_list = []
        for question_info in questions:
            variable_name = question_info.get('Variable_Name')
            if not variable_name:
                continue

            # Make a copy of the original question data to preserve all fields
            new_q_data = question_info.copy()

            # Get score_range and human response
            score_range = question_info.get('score_range')
            human_avg = question_info.get('Average_Human_Response')

            # Fill Average_LLM_Response from CSV
            if variable_name in df.columns:
                numeric_responses = pd.to_numeric(df[variable_name], errors='coerce')
                llm_avg = numeric_responses.dropna().mean()
                new_q_data['Average_LLM_Response'] = round(llm_avg, 4) if pd.notna(llm_avg) else None
            else:
                new_q_data['Average_LLM_Response'] = None

            # Add normalized versions if we have score_range
            if score_range and isinstance(score_range, list) and len(score_range) == 2:
                # Normalize human response
                if human_avg is not None:
                    normalized_human = normalize(human_avg, score_range)
                    new_q_data['Average_Human_Response_norm'] = round(normalized_human, 4)
                else:
                    new_q_data['Average_Human_Response_norm'] = None

                # Normalize LLM response
                llm_avg = new_q_data.get('Average_LLM_Response')
                if llm_avg is not None:
                    normalized_llm = normalize(llm_avg, score_range)
                    new_q_data['Average_LLM_Response_norm'] = round(normalized_llm, 4)
                else:
                    new_q_data['Average_LLM_Response_norm'] = None
            else:
                new_q_data['Average_Human_Response_norm'] = None
                new_q_data['Average_LLM_Response_norm'] = None

            new_question_list.append(new_q_data)

        final_data[category] = new_question_list

    # --- Save Final JSON ---
    print("\nProcessing complete.")
    
    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved data with LLM responses and normalized values to {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"Error: Failed to save the final JSON file. {e}")


if __name__ == '__main__':
    process_and_normalize_final()
