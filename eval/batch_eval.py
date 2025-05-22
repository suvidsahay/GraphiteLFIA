#!/usr/bin/env python3
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import combinations
from typing import Dict, List, Tuple

from model import PrometheusEvaluator  # your Prometheus wrapper

# ─── Ranking Metrics ───────────────────────────────────────────────────────────
def calculate_ranking_metrics(rankings: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """Count how many times each article type lands in each rank position."""
    article_types = set()
    for rank_list in rankings.values():
        article_types.update(rank_list)
    article_types = list(article_types)
    num_places = max(len(rank_list) for rank_list in rankings.values())

    place_counts = [{atype: 0 for atype in article_types} for _ in range(num_places)]
    place_labels = [f"{i+1}{'st' if i==0 else 'nd' if i==1 else 'rd' if i==2 else 'th'} Place"
                    for i in range(num_places)]

    for rank_list in rankings.values():
        for i, atype in enumerate(rank_list):
            place_counts[i][atype] += 1

    return dict(zip(place_labels, place_counts))


def visualize_rankings(metrics: Dict[str, Dict[str, int]], save_path: str):
    """Bar chart of how often each type appears in each rank."""
    article_types = list(next(iter(metrics.values())).keys())
    positions     = list(metrics.keys())
    n_types       = len(article_types)
    bar_width     = 0.8 / n_types
    r_base        = np.arange(len(positions))

    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, atype in enumerate(article_types):
        counts = [metrics[pos][atype] for pos in positions]
        ax.bar(r_base + idx*bar_width, counts, width=bar_width, label=atype.capitalize())

    ax.set_xticks(r_base + bar_width*(n_types-1)/2)
    ax.set_xticklabels(positions)
    ax.set_xlabel("Ranking Position")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Article Types by Ranking Position")
    ax.legend()

    # annotate
    for bar_group in ax.containers:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(f"{int(h)}",
                        xy=(bar.get_x()+bar.get_width()/2, h),
                        xytext=(0,3), textcoords="offset points",
                        ha="center", va="bottom")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Ranking visualization saved to {save_path}")


# ─── Preference Matrix ────────────────────────────────────────────────────────
def calculate_preference_matrix(results_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise count of wins: rows beat columns."""
    article_types = pd.unique(results_df[['Response_A_Type','Response_B_Type']].values.ravel())
    matrix = pd.DataFrame(0, index=article_types, columns=article_types)

    for _, row in results_df.iterrows():
        if row['Score'] == 'A':
            winner, loser = row['Response_A_Type'], row['Response_B_Type']
        else:
            winner, loser = row['Response_B_Type'], row['Response_A_Type']
        matrix.loc[winner, loser] += 1

    return matrix


def visualize_preference_matrix(matrix: pd.DataFrame, save_path: str):
    """Heatmap of the pairwise preference counts."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd', square=True,
                cbar_kws={'label':'Number of Wins'})
    plt.title("Pairwise Preference Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Preference matrix saved to {save_path}")


# ─── Preference Rankings ──────────────────────────────────────────────────────
def analyze_preferences(results_df: pd.DataFrame) -> Dict[str, List[str]]:
    """For each instruction, rank article types by win counts."""
    rankings = {}
    for instr in results_df['Instruction'].unique():
        subset = results_df[results_df['Instruction']==instr]
        wins = {}
        for _, row in subset.iterrows():
            winner = row['Response_A_Type'] if row['Score']=='A' else row['Response_B_Type']
            wins[winner] = wins.get(winner, 0) + 1
        # sort descending
        ranked = [atype for atype,_ in sorted(wins.items(), key=lambda x:x[1], reverse=True)]
        rankings[instr] = ranked
    return rankings


# ─── Core Evaluation ─────────────────────────────────────────────────────────
def evaluate_rubric(
    evaluator: PrometheusEvaluator,
    instructions: List[str],
    responses_A: List[str],
    responses_B: List[str],
    types_A:     List[str],
    types_B:     List[str],
    rubric:      str,
    rubric_name: str
) -> Tuple[pd.DataFrame,pd.DataFrame,Dict[str,List[str]],Dict[str,Dict[str,int]]]:
    """Run A/B on every pair, save CSVs and visualizations."""
    print(f"\n=== Evaluating rubric: {rubric_name} ===")
    rows = []
    for instr, A, B, tA, tB in zip(instructions, responses_A, responses_B, types_A, types_B):
        feedback, score = evaluator.evaluate_pair(instr, A, B, rubric)
        rows.append({
            'Instruction':      instr,
            'Response_A_Type':  tA,
            'Response_B_Type':  tB,
            'Score':            score
        })

    df = pd.DataFrame(rows)
    outdir = f'results/{rubric_name}'
    os.makedirs(outdir, exist_ok=True)

    df.to_csv(f'{outdir}/relative_evaluation_results.csv', index=False)
    print(f"Saved results to {outdir}/relative_evaluation_results.csv")

    pm = calculate_preference_matrix(df)
    pm.to_csv(f'{outdir}/preference_matrix.csv')
    visualize_preference_matrix(pm, f'{outdir}/preference_matrix.png')

    rankings = analyze_preferences(df)
    rk_df   = pd.DataFrame({
        'Instruction': list(rankings.keys()),
        'Ranking':     [', '.join(r) for r in rankings.values()]
    })
    rk_df.to_csv(f'{outdir}/article_rankings.csv', index=False)
    print(f"Saved rankings to {outdir}/article_rankings.csv")

    metrics = calculate_ranking_metrics(rankings)
    met_df  = pd.DataFrame(metrics)
    met_df.to_csv(f'{outdir}/ranking_metrics.csv')
    print(f"Saved ranking metrics to {outdir}/ranking_metrics.csv")

    visualize_rankings(metrics, f'{outdir}/ranking_visualization.png')

    return df, pm, rankings, metrics


# ─── Main ▶︎ build pairs, loop rubrics ───────────────────────────────────────
def main():
    evaluator = PrometheusEvaluator()
    df = pd.read_csv('../data/articles.csv')
    print("Columns:", df.columns.tolist())

    types = ['human_written','generated','unengaging','storm','agent']
    pairs = list(combinations(types, 2))

    instrs, respA, respB, tA, tB = [], [], [], [], []
    template = "{topic}"

    for _, row in df.iterrows():
        topic = row['topic']
        articles = {
            'human_written': row['human-writen article cleaned'],
            'generated':      row['generated article'],
            'unengaging':     row['unengaging generated article'],
            'storm':          row['storm article'],
            'agent':          row['agent article']
        }
        for a, b in pairs:
            instrs.append(template.format(topic=topic))
            respA.append(articles[a])
            respB.append(articles[b])
            tA.append(a)
            tB.append(b)

    rubrics = {
        'engaging':     "Is the article engaging and likely to hold the reader's attention?",
        'fun':          "Is the article fun?",
        'conversational': "Does the article have a more conversational tone?",
        'personality':  "Does the article have personality?",
        'non_repetitive':"Does the article have less repetitive sentence structures?",
        'practical_tips':"Does the article have practical tips?",
        'consistent':   "Is the article consistent?"
    }

    # run and collect
    all_results = {}
    for name, text in rubrics.items():
        res = evaluate_rubric(
            evaluator,
            instrs, respA, respB, tA, tB,
            text, name
        )
        all_results[name] = res

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
