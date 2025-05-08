import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import time
import random
import re
from tqdm import tqdm
import google.generativeai as genai
from typing import List, Tuple
from itertools import combinations


class GeminiEvaluator:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        genai.configure(api_key="")  # Replace with your API key
        self.model_name = model_name
        self.client = genai.GenerativeModel(model_name,
                                   generation_config=genai.GenerationConfig(
                                       temperature=0,
                                   ))

    def _create_prompt(self, instruction: str, response_a: str, response_b: str, criteria: str) -> str:
        prompt_prometheus = f"""###Task Description:
An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), a reference answer, and an evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the two responses strictly based on the given evaluation criteria.
2. Compare Response A and Response B directly.
3. After writing the feedback, indicate the better response ("A" or "B").
4. The format must be: "Feedback: <feedback> [RESULT] <A/B>".

###Instruction:
{instruction}

###Response A:
{response_a}

###Response B:
{response_b}

###Score Rubric:
{criteria}

###Feedback:"""
        return prompt_prometheus

    def extract_result(self, feedback: str):
        match = re.search(r'\[RESULT\]\s*([AB])', feedback)
        if match:
            return feedback, match.group(1)
        match = re.search(r'\[([AB])\]', feedback)
        if match:
            return feedback, match.group(1)
        raise ValueError("Could not extract result (A or B) from feedback.")

    def evaluate_pair(self, instruction: str, response_a: str, response_b: str, criteria: str) -> Tuple[str, str]:
        prompt = self._create_prompt(instruction, response_a, response_b, criteria)
        max_retries = 5
        base_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(prompt)
                feedback = response.text
                feedback, result = self.extract_result(feedback)
                return feedback, result
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print("Max retries exceeded.")
                    raise e

    def evaluate_batch(self, instructions: List[str], responses_a: List[str],
                       responses_b: List[str], criteria: str) -> Tuple[List[str], List[str]]:
        feedbacks, scores = [], []
        for instruction, response_a, response_b in tqdm(zip(instructions, responses_a, responses_b),
                                                        total=len(instructions),
                                                        desc="Evaluating Pairs"):
            feedback, score = self.evaluate_pair(instruction, response_a, response_b, criteria)
            feedbacks.append(feedback)
            scores.append(score)
        return feedbacks, scores


def calculate_ranking_metrics(rankings):
    article_types = set()
    for rank_list in rankings.values():
        article_types.update(rank_list)
    article_types = list(article_types)
    num_places = max(len(rank_list) for rank_list in rankings.values())

    place_counts = [{atype: 0 for atype in article_types} for _ in range(num_places)]

    for ranking in rankings.values():
        for i, article_type in enumerate(ranking):
            place_counts[i][article_type] += 1

    place_labels = [f"{i+1}{'st' if i==0 else 'nd' if i==1 else 'rd' if i==2 else 'th'} Place" for i in range(num_places)]
    metrics = dict(zip(place_labels, place_counts))

    return metrics


def visualize_rankings(metrics, save_path='results/ranking_visualization.png'):
    article_types = list(next(iter(metrics.values())).keys())
    positions = list(metrics.keys())

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.8 / len(article_types)
    r_base = np.arange(len(positions))
    rs = [r_base + i * bar_width for i in range(len(article_types))]

    bars = []
    for idx, atype in enumerate(article_types):
        heights = [metrics[pos][atype] for pos in positions]
        bar = ax.bar(rs[idx], heights, width=bar_width, label=atype.capitalize())
        bars.append(bar)

    ax.set_xlabel('Ranking Position')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Article Types by Ranking Position')
    ax.set_xticks(r_base + bar_width * (len(article_types) - 1) / 2)
    ax.set_xticklabels(positions)
    ax.legend()

    def add_labels(bar_group):
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    for bar_group in bars:
        add_labels(bar_group)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nRanking visualization saved to {save_path}")


def calculate_preference_matrix(results_df):
    article_types = pd.unique(results_df[['Response_A_Type', 'Response_B_Type']].values.ravel())
    preference_matrix = pd.DataFrame(0, index=article_types, columns=article_types)

    for _, row in results_df.iterrows():
        if row['Score'] == 'A':
            winner = row['Response_A_Type']
            loser = row['Response_B_Type']
        else:
            winner = row['Response_B_Type']
            loser = row['Response_A_Type']
        preference_matrix.loc[winner, loser] += 1

    return preference_matrix


def visualize_preference_matrix(matrix, save_path='results/preference_matrix.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd', square=True, cbar_kws={'label': 'Number of Wins'})
    plt.title('Pairwise Preference Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nPreference matrix visualization saved to {save_path}")


def analyze_preferences(results_df):
    rankings = {}
    article_types = pd.unique(results_df[['Response_A_Type', 'Response_B_Type']].values.ravel())

    for instruction in results_df['Instruction'].unique():
        instruction_data = results_df[results_df['Instruction'] == instruction]
        wins = {atype: 0 for atype in article_types}

        for _, row in instruction_data.iterrows():
            if row['Score'] == 'A':
                wins[row['Response_A_Type']] += 1
            elif row['Score'] == 'B':
                wins[row['Response_B_Type']] += 1

        ranked_articles = sorted(wins.items(), key=lambda x: x[1], reverse=True)
        rankings[instruction] = [atype for atype, _ in ranked_articles]

    return rankings


def evaluate_rubric(evaluator, instructions, responses_A, responses_B, response_A_types, response_B_types, rubric, rubric_name):
    print(f"\nEvaluating articles based on rubric: {rubric}")
    feedbacks, scores = evaluator.evaluate_batch(instructions, responses_A, responses_B, rubric)

    results_df = pd.DataFrame({
        'Instruction': instructions,
        'Response_A_Type': response_A_types,
        'Response_B_Type': response_B_types,
        'Score': scores
    })

    rubric_dir = f'results/{rubric_name}'
    os.makedirs(rubric_dir, exist_ok=True)

    results_df.to_csv(f'{rubric_dir}/relative_evaluation_results.csv', index=False)
    print(f"Evaluation completed for {rubric_name}. Results saved.")

    preference_matrix = calculate_preference_matrix(results_df)
    preference_matrix.to_csv(f'{rubric_dir}/preference_matrix.csv')
    visualize_preference_matrix(preference_matrix, f'{rubric_dir}/preference_matrix.png')

    rankings = analyze_preferences(results_df)
    rankings_df = pd.DataFrame({
        'Instruction': list(rankings.keys()),
        'Ranking': [', '.join(rank) for rank in rankings.values()]
    })
    rankings_df.to_csv(f'{rubric_dir}/article_rankings.csv', index=False)

    metrics = calculate_ranking_metrics(rankings)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'{rubric_dir}/ranking_metrics.csv')

    visualize_rankings(metrics, f'{rubric_dir}/ranking_visualization.png')

    return results_df, preference_matrix, rankings, metrics


def main():
    evaluator = GeminiEvaluator()

    df = pd.read_csv('data/article_dataset.csv')
    print("Column names:", df.columns.tolist())

    article_types = ['human_written', 'generated', 'unengaging']
    article_pairs = list(combinations(article_types, 2))

    instructions, responses_A, responses_B, response_A_types, response_B_types = [], [], [], [], []
    instruction_template = "{topic}"

    for _, row in df.iterrows():
        topic = row['topic']
        articles = {
            'human_written': row['human-writen article'],
            'generated': row['generated article'],
            'unengaging': row['unengaging generated article'],
        }

        for article_type_A, article_type_B in article_pairs:
            instructions.append(instruction_template.format(topic=topic))
            responses_A.append(articles[article_type_A])
            responses_B.append(articles[article_type_B])
            response_A_types.append(article_type_A)
            response_B_types.append(article_type_B)

    rubrics = {
        'engaging': "Which article is more fun?",
    }

    for rubric_name, rubric in rubrics.items():
        evaluate_rubric(evaluator, instructions, responses_A, responses_B,
                        response_A_types, response_B_types, rubric, rubric_name)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
