import argparse
import json

from .document_trim import parse_article, assemble_article, trim_article
from .interest_eval import InterestEvaluator
from .coverage_eval import CoverageEvaluator
from .coherence_eval import CoherenceEvaluator
from .relevance_eval import RelevanceEvaluator
from .rouge import RougeEvaluator

class CombinedEvaluator:
    def __init__(self):
        self.interest = InterestEvaluator()
        self.coverage = CoverageEvaluator()
        self.coherence = CoherenceEvaluator()
        self.relevance = RelevanceEvaluator()
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a document using different evaluation methods.")
    parser.add_argument("--document", required=True, help="Path to the document to be evaluated")
    parser.add_argument("--reference", required=True, help="Path to the reference document")
    parser.add_argument("--instruction", required=True, help="Instruction string used to generate the document")
    parser.add_argument("--EvaluationMethod", default="all",
                        choices=["prom_interest", "prom_coverage", "prom_coherence", "prom_relevance", "ROUGE", "all"],
                        help="Evaluation method to use")

    args = parser.parse_args()

    with open(args.document, "r", encoding="utf-8") as f:
        document = f.read()
    with open(args.reference, "r", encoding="utf-8") as f:
        reference = f.read()

    evaluator = CombinedEvaluator()

    if args.EvaluationMethod == "all":
        result = evaluator.evaluate_all(args.instruction, document, reference)
    else:
        result = evaluator.evaluate_single(args.EvaluationMethod, args.instruction, document, reference)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
