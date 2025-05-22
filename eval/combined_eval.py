import argparse
import json
import os

from .document_trim import parse_article, assemble_article, trim_article
from .interest_eval import InterestEvaluator
from .coverage_eval import CoverageEvaluator
from .coherence_eval import CoherenceEvaluator
from .relevance_eval import RelevanceEvaluator
from .rouge import RougeEvaluator

def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read().strip()

class CombinedEvaluator:
    def __init__(self, model_name="kaist-ai/Prometheus-7b-v1.0", tokenizer_name="meta-llama/Llama-2-7b-chat-hf", device="auto"):
        self.interest = InterestEvaluator(model_name, tokenizer_name, device)
        self.coverage = CoverageEvaluator(model_name, tokenizer_name, device)
        self.coherence = CoherenceEvaluator(model_name, tokenizer_name, device)
        self.relevance = RelevanceEvaluator(model_name, tokenizer_name, device)
        self.rouge = RougeEvaluator()

    def evaluate_all(self, instruction, response, reference):
        document_tree = parse_article(response)
        response_trimmed = response
        if document_tree.total_word_count() > 2000:
            response_trimmed = assemble_article(trim_article(document_tree, 2000))
        return {
            "Interest": self.interest.evaluate_interest(instruction, response_trimmed),
            "Coverage": self.coverage.evaluate_coverage(instruction, response_trimmed),
            "Coherence": self.coherence.evaluate_coherence(instruction, response_trimmed),
            "Relevance": self.relevance.evaluate_relevance(instruction, response_trimmed),
            "ROUGE": self.rouge.calculate_rouge(response, reference)
        }

    def evaluate_single(self, method, instruction, response, reference):
        if method == "prom_interest":
            return {"Interest": self.interest.evaluate_interest(instruction, response)}
        elif method == "prom_coverage":
            return {"Coverage": self.coverage.evaluate_coverage(instruction, response)}
        elif method == "prom_coherence":
            return {"Coherence": self.coherence.evaluate_coherence(instruction, response)}
        elif method == "prom_relevance":
            return {"Relevance": self.relevance.evaluate_relevance(instruction, response)}
        elif method == "ROUGE":
            return {"ROUGE": self.rouge.calculate_rouge(response, reference)}
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

def evaluate_model(evaluator, model_name, folder, filename, aggregated_results, valid_counts):
    response = read_file(os.path.join("data", folder, filename))
    reference = read_file(os.path.join("data/FreshWiki/txt", filename))
    topic = filename.rsplit(".", 1)[0].replace("_", " ")
    instruction = f"Write a 2000 word article on {topic}"

    results = evaluator.evaluate_all(instruction, response, reference)

    print(f"Results for {model_name} {topic}:")
    for category, result in results.items():
        print(f"{category}: {result}")

        if isinstance(result, dict) and 'score' in result:
            score = result['score']
            if score is not None:
                aggregated_results[category] += score
                valid_counts[category] += 1

        elif isinstance(result, dict):
            for rouge_metric, score_obj in result.items():
                aggregated_results[f"{rouge_metric}_precision"] += score_obj.precision
                aggregated_results[f"{rouge_metric}_recall"] += score_obj.recall
                aggregated_results[f"{rouge_metric}_fmeasure"] += score_obj.fmeasure
                valid_counts[f"{rouge_metric}_precision"] += 1
                valid_counts[f"{rouge_metric}_recall"] += 1
                valid_counts[f"{rouge_metric}_fmeasure"] += 1
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate documents using various methods.")
    parser.add_argument("--document", required=True, help="Path to document or directory")
    parser.add_argument("--reference", required=True, help="Path to reference document or directory")
    parser.add_argument("--instruction", help="Instruction string used to generate the document")
    parser.add_argument("--EvaluationMethod", default="all",
                        choices=["prom_interest", "prom_coverage", "prom_coherence", "prom_relevance", "ROUGE", "all"],
                        help="Evaluation method to use")
    parser.add_argument("--model_name", default="kaist-ai/Prometheus-13b-v1.0")
    parser.add_argument("--tokenizer_name", default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    evaluator = CombinedEvaluator(args.model_name, args.tokenizer_name, args.device)

    if os.path.isdir(args.document):
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
        valid_counts = {key: 0 for key in aggregated_results}
        files = [f for f in os.listdir(args.document) if f.endswith(".txt")]

        for filename in files:
            evaluate_model(evaluator, "CustomModel", os.path.basename(args.document), filename, aggregated_results, valid_counts)

        avg_scores = {
            metric: (aggregated_results[metric] / valid_counts[metric]) if valid_counts[metric] > 0 else None
            for metric in aggregated_results
        }

        print("\nAverage Scores Across All Files:")
        for metric, avg_score in avg_scores.items():
            print(f"{metric}: {avg_score:.4f}" if avg_score is not None else f"{metric}: No valid scores")

    else:
        with open(args.document, "r", encoding="utf-8") as f:
            document = f.read()
        with open(args.reference, "r", encoding="utf-8") as f:
            reference = f.read()

        instruction = args.instruction or f"Write a 2000 word article on {os.path.splitext(os.path.basename(args.document))[0].replace('_', ' ')}"

        if args.EvaluationMethod == "all":
            result = evaluator.evaluate_all(instruction, document, reference)
        else:
            result = evaluator.evaluate_single(args.EvaluationMethod, instruction, document, reference)

        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
