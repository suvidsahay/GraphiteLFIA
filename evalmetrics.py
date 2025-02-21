from eval import InterestEvaluator
from evalbc import CoverageEvaluator
from evalco import CoherenceEvaluator
from evalrf import RelevanceEvaluator
#from .rouge import compute_rouge  

class CombinedEvaluator:
    def __init__(self):
        self.interest = InterestEvaluator()
        self.coverage = CoverageEvaluator()
        self.coherence = CoherenceEvaluator()
        self.relevance = RelevanceEvaluator()

    def evaluate_all(self, instruction, response, reference):
        return {
            "Interest": self.interest.evaluate_interest(instruction, response),
            "Coverage": self.coverage.evaluate_coverage(instruction, response),
            "Coherence": self.coherence.evaluate_coherence(instruction, response),
            "Relevance": self.relevance.evaluate_relevance(instruction, response),
            #"ROUGE": compute_rouge(response, reference)
        }