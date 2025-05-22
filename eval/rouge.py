from rouge_score import rouge_scorer

class RougeEvaluator:
    def __init__(self, rouge_types=['rouge1', 'rouge2', 'rougeL']):
        """
        Initializes the RougeEvaluator with specified ROUGE types.

        Args:
        rouge_types: A list of ROUGE types to calculate (default: ['rouge1', 'rouge2', 'rougeL']).
        """
        self.rouge_types = rouge_types
        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)

    def calculate_rouge(self, reference, candidate):
        """
        Calculates ROUGE scores for a given reference and candidate summary.

        Args:
        reference: The reference summary (string).
        candidate: The candidate summary (string).

        Returns:
        A dictionary containing the ROUGE scores for each type.
        """
        return self.scorer.score(reference, candidate)


if __name__ == "__main__":
    # Example Usage
    generated = "The military offers diverse careers, including engineering and medicine."
    reference = "A military career provides opportunities in fields like medicine and engineering."

    evaluator = RougeEvaluator()
    rouge_scores = evaluator.calculate_rouge(reference, generated)

    print("ROUGE Scores:", rouge_scores)
