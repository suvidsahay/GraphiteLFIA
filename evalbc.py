from baseevaluator import BaseEvaluator

class CoverageEvaluator(BaseEvaluator):
    rubric = "[Broad Coverage: Does the article provide an in-depth exploration of the topic and have good coverage?] \
                Score 1: Severely lacking; offers little to no coverage of the topic’s primary aspects, resulting in a very narrow perspective. \
                Score 2: Partial coverage; includes some of the topic’s main aspects but misses others, resulting in an incomplete portrayal. \
                Score 3: Acceptable breadth; covers most main aspects, though it may stray into minor unnecessary details or overlook some relevant points. \
                Score 4: Good coverage; achieves broad coverage of the topic, hitting on all major points with minimal extraneous information. \
                Score 5: Exemplary in breadth; delivers outstanding coverage, thoroughly detailing all crucial aspects of the topic without including irrelevant information."


    def evaluate_coverage(self, instruction, response):
        return self.evaluate(self.rubric, instruction, response)