import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import re
class BaseEvaluator:
    def __init__(self, model_name="kaist-ai/Prometheus-7b-v1.0", tokenizer_name="meta-llama/Llama-2-7b-chat-hf", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name, device_map=device)

    def evaluate(self, rubric, instruction, response):
        input_text = f"###Task Description: An instruction, a response, and a rubric are given. Assess the response based on the rubric and return feedback and a score (1-5). Format: 'Feedback: (feedback) [RESULT] (score)'. ###Instruction: {instruction} ###Response: {response} ###Rubric: {rubric} ###Feedback:"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids, do_sample=True, temperature=1.0, top_p=0.9, max_new_tokens=256, repetition_penalty=1.03)
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        match = re.search(r'Feedback:(.*?)\[\s*RESULT\s*\]\s*(\d+)', decoded_output, re.DOTALL)
        if match:
            return {"feedback": match.group(1).strip(), "score": int(match.group(2))}
        return {"feedback": "Error parsing response", "score": None}