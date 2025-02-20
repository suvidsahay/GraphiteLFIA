from rouge_score import rouge_scorer

def calculate_rouge(reference, candidate, rouge_types=['rouge1', 'rouge2', 'rougeL']):
    """
    Calculates ROUGE scores for a given reference and candidate summary.

    Args:
    reference: The reference summary (string).
    candidate: The candidate summary (string).
    rouge_types: A list of ROUGE types to calculate (e.g., ['rouge1', 'rouge2', 'rougeL']).

    Returns:
    A dictionary containing the ROUGE scores for each type.
    """
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores


if __name__ == "__main__":
    # Example Usage
    generated = "The military offers diverse careers, including engineering and medicine."
    reference = "A military career provides opportunities in fields like medicine and engineering."

    rouge_scores = calculate_rouge(generated, reference)
    print("ROUGE Scores:", rouge_scores)
