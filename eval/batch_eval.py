import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import seaborn as sns
import pandas as pd
import os
from model import PrometheusEvaluator, GeminiEvaluator

def calculate_ranking_metrics(rankings):
    """Calculate metrics about article type rankings."""
    # Initialize counters for each position
    first_place = {'human_written': 0, 'generated': 0, 'unengaging': 0, 'storm': 0, 'agent': 0}
    second_place = {'human_written': 0, 'generated': 0, 'unengaging': 0, 'storm': 0, 'agent': 0}
    third_place = {'human_written': 0, 'generated': 0, 'unengaging': 0, 'storm': 0, 'agent': 0}
    fourth_place = {'human_written': 0, 'generated': 0, 'unengaging': 0, 'storm': 0, 'agent': 0}
    fifth_place = {'human_written': 0, 'generated': 0, 'unengaging': 0, 'storm': 0, 'agent': 0}
    
    # Count occurrences in each position
    for ranking in rankings.values():
        first_place[ranking[0]] += 1
        second_place[ranking[1]] += 1
        third_place[ranking[2]] += 1
        fourth_place[ranking[3]] += 1
        fifth_place[ranking[4]] += 1
    
    # Return raw counts
    metrics = {
        'First Place': first_place,
        'Second Place': second_place,
        'Third Place': third_place,
        'Fourth Place': fourth_place,
        'Fifth Place': fifth_place
    }
    
    return metrics

def visualize_rankings(metrics, save_path='../results/ranking_visualization.png'):
    """Create a bar graph visualization of ranking metrics."""
    # Prepare data
    article_types = ['human_written', 'generated', 'unengaging', 'storm', 'agent']
    positions = ['First Place', 'Second Place', 'Third Place', 'Fourth Place', 'Fifth Place']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.15
    
    # Set position of bars on X axis
    r1 = np.arange(len(positions))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    
    # Create bars for each article type
    bars1 = ax.bar(r1, [metrics[pos]['human_written'] for pos in positions], 
                  width=bar_width, label='Human Written', color='#2ecc71')
    bars2 = ax.bar(r2, [metrics[pos]['generated'] for pos in positions], 
                  width=bar_width, label='Generated', color='#3498db')
    bars3 = ax.bar(r3, [metrics[pos]['unengaging'] for pos in positions], 
                  width=bar_width, label='Unengaging', color='#e74c3c')
    bars4 = ax.bar(r4, [metrics[pos]['storm'] for pos in positions], 
                  width=bar_width, label='Storm', color='#9b59b6')
    bars5 = ax.bar(r5, [metrics[pos]['agent'] for pos in positions], 
                  width=bar_width, label='Agent', color='#f1c40f')
    
    # Add labels and title
    ax.set_xlabel('Ranking Position')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Article Types by Ranking Position')
    ax.set_xticks([r + 2*bar_width for r in range(len(positions))])
    ax.set_xticklabels(positions)
    
    # Add legend
    ax.legend()
    
    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)
    add_labels(bars5)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nRanking visualization saved to {save_path}")

def calculate_preference_matrix(results_df):
    """Calculate a matrix showing how many times each article type was preferred over others."""
    article_types = ['human_written', 'generated', 'unengaging', 'storm', 'agent']
    preference_matrix = pd.DataFrame(0, index=article_types, columns=article_types)
    
    for _, row in results_df.iterrows():
        if row['Score'] == 'A':  # Response_A was preferred
            winner = row['Response_A_Type']
            loser = row['Response_B_Type']
        else:  # Response_B was preferred
            winner = row['Response_B_Type']
            loser = row['Response_A_Type']
        
        preference_matrix.loc[winner, loser] += 1
    
    return preference_matrix

def visualize_preference_matrix(matrix, save_path='../results/preference_matrix.png'):
    """Create a heatmap visualization of the preference matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd', 
                square=True, cbar_kws={'label': 'Number of Wins'})
    plt.title('Pairwise Preference Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nPreference matrix visualization saved to {save_path}")

def analyze_preferences(results_df):
    # Group by instruction and analyze preferences
    rankings = {}
    for instruction in results_df['Instruction'].unique():
        instruction_data = results_df[results_df['Instruction'] == instruction]
        
        # Create a dictionary to track wins for each article type
        article_types = ['human_written', 'generated', 'unengaging', 'storm', 'agent']
        wins = {article: 0 for article in article_types}
        
        # Count wins for each article type
        for _, row in instruction_data.iterrows():
            if row['Score'] == 'A':  # Response_A was preferred
                wins[row['Response_A_Type']] += 1
            elif row['Score'] == 'B':  # Response_B was preferred
                wins[row['Response_B_Type']] += 1
        
        # Sort article types by number of wins (descending)
        ranked_articles = sorted(wins.items(), key=lambda x: x[1], reverse=True)
        rankings[instruction] = [article for article, _ in ranked_articles]
    
    return rankings

def evaluate_rubric(evaluator, instructions, responses_A, responses_B, response_A_types, response_B_types, rubric, rubric_name):
    """Evaluate articles against a specific rubric and save results."""
    print(f"\nEvaluating articles based on rubric: {rubric}")
    
    feedbacks, scores = evaluator.evaluate_batch(
        instructions=instructions,
        responses_a=responses_A,
        responses_b=responses_B,
        criteria=rubric
    )

    # Create DataFrame with article types instead of full content
    results_df = pd.DataFrame({
        'Instruction': instructions,
        'Response_A_Type': response_A_types,
        'Response_B_Type': response_B_types,
        'Score': scores
    })
    
    # Create directory for rubric-specific results if it doesn't exist
    rubric_dir = f'../results/{rubric_name}'
    os.makedirs(rubric_dir, exist_ok=True)
    
    # Save the comparison results
    results_df.to_csv(f'{rubric_dir}/relative_evaluation_results.csv', index=False)
    print(f"Evaluation completed for {rubric_name}. Results saved.")
    
    # Calculate and display preference matrix
    preference_matrix = calculate_preference_matrix(results_df)
    print(f"\nPairwise Preference Matrix for {rubric_name}:")
    print(preference_matrix)
    
    # Save preference matrix
    preference_matrix.to_csv(f'{rubric_dir}/preference_matrix.csv')
    print(f"\nPreference matrix saved to {rubric_dir}/preference_matrix.csv")
    
    # Visualize preference matrix
    visualize_preference_matrix(preference_matrix, f'{rubric_dir}/preference_matrix.png')
    
    # Analyze preferences and generate rankings
    rankings = analyze_preferences(results_df)
    
    # Create a DataFrame for rankings
    rankings_df = pd.DataFrame({
        'Instruction': list(rankings.keys()),
        'Ranking': [', '.join(rank) for rank in rankings.values()]
    })
    
    # Save rankings to a separate file
    rankings_df.to_csv(f'{rubric_dir}/article_rankings.csv', index=False)
    print(f"\nArticle rankings saved to {rubric_dir}/article_rankings.csv")
    
    # Calculate and save ranking metrics
    metrics = calculate_ranking_metrics(rankings)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'{rubric_dir}/ranking_metrics.csv')
    print(f"\nRanking metrics saved to {rubric_dir}/ranking_metrics.csv")
    
    # Create and save visualization
    visualize_rankings(metrics, f'{rubric_dir}/ranking_visualization.png')
    
    return results_df, preference_matrix, rankings, metrics

def main():
    # Choose which evaluator to use
    evaluator_type = "gemini"  # or "prometheus"
    
    if evaluator_type == "gemini":
        evaluator = GeminiEvaluator()
    else:
        evaluator = PrometheusEvaluator()

    df = pd.read_csv('../data/articles.csv')
    print("Column names in DataFrame:", df.columns.tolist())
    print("First row:", df.iloc[0].to_dict())

    instructions, responses_A, responses_B = [], [], []
    response_A_types, response_B_types = [], []
    instruction_template = "{topic}"

    # Get all possible pairs of article types
    article_pairs = list(combinations(['human_written', 'generated', 'unengaging', 'storm', 'agent'], 2))

    for _, row in df.iterrows():
        topic = row['topic']
        articles = {
            'human_written': row['human-writen article cleaned'],
            'generated': row['generated article'],
            'unengaging': row['unengaging generated article'],
            'storm': row['storm article'],
            'agent': row['agent article']
        }

        # Add all possible pairwise comparisons for each row
        for article_type_A, article_type_B in article_pairs:
            instructions.append(instruction_template.format(topic=topic))
            responses_A.append(articles[article_type_A])
            responses_B.append(articles[article_type_B])
            response_A_types.append(article_type_A)
            response_B_types.append(article_type_B)

    # Define rubrics
    rubrics = {
        'engaging': "Is the article engaging and likely to hold the reader's attention?",
        'fun': "Is the article fun?",
        'conversational': "Does the article have a more conversational tone?",
        'personality': "Does the article have personality?",
        'non_repetitive': "Does the article have less repetitive sentence structures?",
        'practical_tips': "Does the article have any practical tips?",
        'consistent': "Is the article consistent?"
    }

    # Evaluate each rubric
    all_results = {}
    for rubric_name, rubric in rubrics.items():
        print(f"\n{'='*50}")
        print(f"Evaluating rubric: {rubric_name}")
        print(f"{'='*50}")
        
        results_df, preference_matrix, rankings, metrics = evaluate_rubric(
            evaluator, instructions, responses_A, responses_B, 
            response_A_types, response_B_types, rubric, rubric_name
        )
        
        all_results[rubric_name] = {
            'results_df': results_df,
            'preference_matrix': preference_matrix,
            'rankings': rankings,
            'metrics': metrics
        }

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

