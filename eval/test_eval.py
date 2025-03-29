import os
from .combined_eval import CombinedEvaluator


def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read().strip()


def evaluate_model(evaluator, model_name, folder, filename, aggregated_results, valid_counts):
    response = read_file(os.path.join("eval/test", folder, filename))
    reference = read_file(os.path.join("data/FreshWiki/txt", filename))

    topic = filename.rsplit(".", 1)[0].replace("_", " ")
    instruction = f"Write an article on {topic}"

    results = evaluator.evaluate_all(instruction, response, reference)

    print(f"Results for {model_name} {topic}:")
    for category, result in results.items():
        print(f"{category}: {result}")

        # Handle 'score' values properly
        if isinstance(result, dict) and 'score' in result:
            score = result['score']
            if score is not None:
                aggregated_results[category] += score
                valid_counts[category] += 1  # Count valid scores

        elif isinstance(result, dict):  # Handle ROUGE scores
            for rouge_metric, score_obj in result.items():
                aggregated_results[f"{rouge_metric}_precision"] += score_obj.precision
                aggregated_results[f"{rouge_metric}_recall"] += score_obj.recall
                aggregated_results[f"{rouge_metric}_fmeasure"] += score_obj.fmeasure

                valid_counts[f"{rouge_metric}_precision"] += 1
                valid_counts[f"{rouge_metric}_recall"] += 1
                valid_counts[f"{rouge_metric}_fmeasure"] += 1

    print("-" * 50)


def main():
    evaluator = CombinedEvaluator()

    # Dictionary to store summed scores
    aggregated_results = {
        "Interest": 0.0,
        "Coverage": 0.0,
        "Coherence": 0.0,
        "Relevance": 0.0,
        "rouge1_precision": 0.0,
        "rouge1_recall": 0.0,
        "rouge1_fmeasure": 0.0,
        "rouge2_precision": 0.0,
        "rouge2_recall": 0.0,
        "rouge2_fmeasure": 0.0,
        "rougeL_precision": 0.0,
        "rougeL_recall": 0.0,
        "rougeL_fmeasure": 0.0
    }

    # Dictionary to track count of valid scores for each metric
    valid_counts = {key: 0 for key in aggregated_results}

    files = [f for f in os.listdir("eval/test/storm_interface") if
             os.path.isfile(os.path.join("eval/test/storm_interface", f))]


    models = {
        "STORM": "storm_interface",
        # "Zero-shot GPT-4": "zero_shot_gpt_4",
    }

    for filename in files:
        for model_name, folder in models.items():
            evaluate_model(evaluator, model_name, folder, filename, aggregated_results, valid_counts)

    # Compute average scores only where valid counts exist
    avg_scores = {
        metric: (aggregated_results[metric] / valid_counts[metric]) if valid_counts[metric] > 0 else None
        for metric in aggregated_results
    }

    # Print average results
    print("\nAverage Scores Across All Files:")
    for metric, avg_score in avg_scores.items():
        print(f"{metric}: {avg_score:.4f}" if avg_score is not None else f"{metric}: No valid scores")


if __name__ == "__main__":
    main()
