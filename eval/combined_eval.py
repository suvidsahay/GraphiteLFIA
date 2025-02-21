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
        return {
            "Interest": self.interest.evaluate_interest(instruction, response),
            "Coverage": self.coverage.evaluate_coverage(instruction, response),
            "Coherence": self.coherence.evaluate_coherence(instruction, response),
            "Relevance": self.relevance.evaluate_relevance(instruction, response),
            "ROUGE": self.rouge.calculate_rouge(response, reference)
        }