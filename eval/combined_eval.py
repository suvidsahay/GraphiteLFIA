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
        print(response)
        return {
            "Interest": self.interest.evaluate_interest(instruction, response_trimmed),
            "Coverage": self.coverage.evaluate_coverage(instruction, response_trimmed),
            "Coherence": self.coherence.evaluate_coherence(instruction, response_trimmed),
            "Relevance": self.relevance.evaluate_relevance(instruction, response_trimmed),
            "ROUGE": self.rouge.calculate_rouge(response, reference)
        }

