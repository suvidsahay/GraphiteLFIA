from .base_eval import BaseEvaluator

class CoherenceEvaluator(BaseEvaluator):
    rubric = "[Coherence and Organization: Is the article well-organized and logically structured?] \
                Score 1: Disorganized; lacks logical structure and coherence. \
                Score 2: Fairly organized; a basic structure is present but not consistently followed. \
                Score 3: Organized; a clear structure is mostly followed with some lapses in coherence. \
                Score 4: Good organization; a clear structure with minor lapses in coherence \
                Score 5: Excellently organized; the article is logically structured with seamless transitions and a clear argument."

    def __init__(self, model_name="kaist-ai/Prometheus-7b-v1.0", tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
                 device="auto"):
        super().__init__(model_name=model_name, tokenizer_name=tokenizer_name, device=device)

    def evaluate_coherence(self, instruction, response):
        return self.evaluate(self.rubric, instruction, response)