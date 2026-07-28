import pandas as pd
import json
import numpy as np
import re
from collections import defaultdict

def get_category(variable_name, question_text):
    """
    Categorizes a question based on its variable name and text,
    following the structure from the provided Table 1.
    """
    name = variable_name.strip()

    # This mapping is based on Table 1 and inspection of the variable names
    # Keywords are mapped to their main category
    keyword_to_category = {
        # Demographics
        'Demographics': 'Demographics',
        # Personality traits
        'Big Five': 'Personality traits',
        'Big 5': 'Personality traits',
        'Need for cognition': 'Personality traits',
        'Agentic Communal': 'Personality traits',
        'Agentic vs. Communal': 'Personality traits',
        'Minimalism': 'Personality traits',
        'Consumer Minimalism': 'Personality traits',
        'Empathy': 'Personality traits',
        'GREEN': 'Personality traits',
        'Green values': 'Personality traits',
        'Social Desirability': 'Personality traits',
        'Conscientiousness': 'Personality traits',
        'Anxiety': 'Personality traits',
        'Beck Anxiety': 'Personality traits',
        'Individualism': 'Personality traits',
        'Individualism vs. Collectivism': 'Personality traits',
        'Selves questionnaire': 'Personality traits',
        'Regulatory Focus': 'Personality traits',
        'Spendthrift': 'Personality traits',
        'Tightwads vs. Spendthrift': 'Personality traits',
        'Depression': 'Personality traits',
        'Need for uniqueness': 'Personality traits',
        'Self-Monitoring': 'Personality traits',
        'Self-monitoring': 'Personality traits',
        'Self-Conc. Clarity': 'Personality traits',
        'Self-concept clarity': 'Personality traits',
        'Need for closure': 'Personality traits',
        'Maximization': 'Personality traits',
        # Cognitive abilities
        'CRT': 'Cognitive abilities',
        'Cognitive Reflection Test': 'Cognitive abilities',
        'Matrix': 'Cognitive abilities',
        'Fluid intelligence': 'Cognitive abilities',
        '3D q': 'Cognitive abilities',
        'Synonym': 'Cognitive abilities',
        'Antonym': 'Cognitive abilities',
        'Crystallized intelligence': 'Cognitive abilities',
        'Syllogism': 'Cognitive abilities',
        'Overconfidence': 'Cognitive abilities',
        'Overplacement': 'Cognitive abilities',
        'Financial literacy': 'Cognitive abilities',
        'Numeracy': 'Cognitive abilities',
        'Deductive certainty': 'Cognitive abilities',
        'Forward Flow': 'Cognitive abilities',
        'Wason Selection Task': 'Cognitive abilities',
        # Economic preferences - 包含所有可量化的题目
        'Ultimatum': 'Economic preferences',
        'Mental accounting': 'Economic preferences',
        'Mental Accounting': 'Economic preferences',
        'Discount': 'Economic preferences',
        'Present bias': 'Economic preferences',
        'Risk Aversion': 'Economic preferences',
        'Loss Aversion': 'Economic preferences',
        'Loss aversion': 'Economic preferences',
        'Trust': 'Economic preferences',
        'Dictator': 'Economic preferences',
        # Heuristics and biases
        'Base rate': 'Heuristics and biases',
        'Outcome bias': 'Heuristics and biases',
        'Sunk cost': 'Heuristics and biases',
        'Sunk cost fallacy': 'Heuristics and biases',
        'Allais': 'Heuristics and biases',
        'Framing': 'Heuristics and biases',
        'Linda': 'Heuristics and biases',
        'Conjunction problem': 'Heuristics and biases',
        'Anchoring': 'Heuristics and biases',
        'Anchoring and adjustment': 'Heuristics and biases',
        'Absolute vs. relative': 'Heuristics and biases',
        'Absolute vs. relative savings': 'Heuristics and biases',
        'Myside': 'Heuristics and biases',
        'Myside bias': 'Heuristics and biases',
        'Less is More': 'Heuristics and biases',
        'WTA/WTP': 'Heuristics and biases',
        'WTA/WTP -- Thaler': 'Heuristics and biases',
        'False Cons.': 'Heuristics and biases',
        'False consensus': 'Heuristics and biases',
        'nonseparability': 'Heuristics and biases',
        'Nonseparability': 'Heuristics and biases',
        'Nonseparability of risk': 'Heuristics and biases',
        'Omission bias': 'Heuristics and biases',
        'Probability matching': 'Heuristics and biases',
        'Probability matching vs. maximizing': 'Heuristics and biases',
        'Dominator neglect': 'Heuristics and biases',
        # Product preferences
        'Pricing': 'Product preferences',
        'choice questions': 'Product preferences',
    }

    # Check for keywords in the variable name first
    for keyword, category in keyword_to_category.items():
        if keyword.lower() in name.lower():
            return category

    # Check for keywords in the question text as a fallback
    for keyword, category in keyword_to_category.items():
        if keyword.lower() in question_text.lower():
            return category

    # Handle Demographics Q...
    if re.match(r'^Q\d+', name):
        return "Demographics"

    return "Uncategorized"

def find_and_categorize_likert_questions():
    """
    Identifies Likert-scale questions, including binary choices that can be quantified,
    categorizes them based on Table 1 and extended mappings, and saves the result.
    """
    categorized_results = defaultdict(list)
    processed_questions = set()

    for i in range(1, 5):
        labels_file = f'data/wave_csv/wave_{i}_labels_anonymized.csv'
        numbers_file = f'data/wave_csv/wave_{i}_numbers_anonymized.csv'

        try:
            labels_df = pd.read_csv(labels_file, header=0, low_memory=False)
            numbers_df = pd.read_csv(numbers_file, header=0, low_memory=False)
        except FileNotFoundError:
            print(f"Wave {i} files not found. Skipping.")
            continue

        questions = labels_df.iloc[0].to_dict()

        for col in numbers_df.columns:
            if col in ['TWIN_ID', 'StartDate', 'EndDate', 'Progress', 'Duration (in seconds)',
                      'Finished', 'RecordedDate', 'pageNo', 'tasktime', 'worktimeArray',
                      'offTask', 'onTask', 'totalOffTask', 'totalOnTask', 'perPagePT',
                      'randDollars', 'randDollarsString']:
                continue

            numeric_col = pd.to_numeric(numbers_df[col], errors='coerce')
            numeric_col.replace([np.inf, -np.inf], np.nan, inplace=True)
            numeric_col.dropna(inplace=True)

            if numeric_col.empty:
                continue

            unique_values = numeric_col.nunique()
            min_val = numeric_col.min()
            max_val = numeric_col.max()

            # 判断是否为整数
            is_integer_scale = (numeric_col.apply(lambda x: np.isfinite(x) and x == np.round(x))).all()

            # 扩展Likert scale定义：
            # 1. 传统多选项量表 (3-11个选项)
            # 2. 特定的二元选择量表 (如Social Desirability)
            # 3. 可量化的二元选择 (如Risk Aversion, Loss Aversion)

            is_likert = False

            # 标准1: 传统多选项量表
            if 3 <= unique_values <= 11 and min_val >= 0 and is_integer_scale:
                is_likert = True

            # 标准2: 特定的二元选择量表
            elif unique_values == 2:
                # 检查是否是已知的量表
                binary_scales = [
                    'Social Desirability',
                    'Risk Aversion', 'Loss Aversion',
                    'Present bias', 'Discount', 'Mental accounting'
                ]
                for scale in binary_scales:
                    if scale.lower() in col.lower():
                        is_likert = True
                        break

            # 标准3: 小范围的连续值（可能是转换后的量表）
            elif unique_values > 2 and unique_values <= 20 and max_val - min_val <= 20:
                # 检查是否可能是百分比或金额量表
                if max_val <= 100 and min_val >= 0:
                    is_likert = True

            if is_likert:
                question_text = questions.get(col, "")

                if question_text and question_text not in processed_questions:
                    avg_response = numeric_col.mean()
                    category = get_category(col, question_text)

                    # 创建scale_range
                    if unique_values == 2:
                        # 二元选择保持原样
                        scale_range = [int(min_val), int(max_val)]
                    else:
                        scale_range = [int(min_val), int(max_val)]

                    mapping_df = pd.DataFrame({'number': numbers_df[col], 'label': labels_df[col]})
                    mapping_df.dropna(subset=['number', 'label'], inplace=True)
                    mapping_df['number'] = pd.to_numeric(mapping_df['number'], errors='coerce')
                    mapping_df.dropna(subset=['number'], inplace=True)
                    mapping_df = mapping_df[mapping_df['number'].apply(lambda x: x == int(x))]
                    mapping_df['number'] = mapping_df['number'].astype(int)

                    scale_map = mapping_df.drop_duplicates().sort_values('number').set_index('number')['label'].to_dict()

                    scale_str = ", ".join([f"{k}: {v}" for k, v in scale_map.items()])

                    enhanced_question = f"{question_text} (Scale: {scale_str})"

                    categorized_results[category].append({
                        "Variable_Name": col,
                        "Question": enhanced_question,
                        "Average_Human_Response": {
                            "value": round(avg_response, 3),
                            "score_range": scale_range
                        }
                    })
                    processed_questions.add(question_text)

    # 输出统计信息
    print("\n分类统计:")
    print("-" * 50)
    total_questions = 0
    for category, questions in categorized_results.items():
        count = len(questions)
        total_questions += count
        print(f"{category}: {count} 题")
    print(f"\n总计: {total_questions} 题")

    # 保存结果
    with open('likert_question_averages_with_scale.json', 'w') as f:
        # 将defaultdict转换为普通dict
        regular_dict = {k: v for k, v in categorized_results.items()}
        json.dump(regular_dict, f, indent=2)

    print(f"\n成功处理并保存 {len(processed_questions)} 个问题到 likert_question_averages_with_scale.json")

if __name__ == "__main__":
    find_and_categorize_likert_questions()