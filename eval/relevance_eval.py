from .base_eval import BaseEvaluator

class RelevanceEvaluator(BaseEvaluator):
    rubric = "[Relevance and Focus: Does the article stay on topic and maintain a clear focus?] \
                Score 1: Off-topic; the content does not align with the headline or core subject. \
                Score 2: Somewhat on topic but with several digressions; the core subject is evident but not consistently adhered to. \
                Score 3: Generally on topic, despite a few unrelated details. \
                Score 4: Mostly on topic and focused; the narrative has a consistent relevance to the core subject with infrequent digressions. \
                Score 5: Exceptionally focused and entirely on topic; the article is tightly centered on the subject, with every piece of information contributing to a comprehensive understanding of the topic."

    def __init__(self, model_name="kaist-ai/Prometheus-7b-v1.0", tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
                 device="auto"):
        super().__init__(model_name=model_name, tokenizer_name=tokenizer_name, device=device)

    def evaluate_relevance(self, instruction, response):
        return self.evaluate(self.rubric, instruction, response)