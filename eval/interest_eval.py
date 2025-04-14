from .base_eval import BaseEvaluator

class InterestEvaluator(BaseEvaluator):
    rubric = "[Interest Level: How engaging and thought-provoking is the article?] \
                Score 1: Not engaging at all; no attempt to capture the reader's attention. \
                Score 2: Fairly engaging with a basic narrative but lacking depth. \
                Score 3: Moderately engaging with several interesting points \
                Score 4: Quite engaging with a well-structured narrative and noteworthy points that frequently capture and retain attention. \
                Score 5: Exceptionally engaging throughout, with a compelling narrative that consistently stimulates interest."
    def __init__(self, model_name="kaist-ai/Prometheus-7b-v1.0", tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
                 device="auto"):
        super().__init__(model_name=model_name, tokenizer_name=tokenizer_name, device=device)

    def evaluate_interest(self, instruction, response):
        return self.evaluate(self.rubric, instruction, response)