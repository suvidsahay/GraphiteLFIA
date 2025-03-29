import logging
import torch
import re
from transformers import AutoTokenizer, LlamaForCausalLM
from model import ModelLoader
from fastchat.conversation import get_conv_template

class BaseEvaluator:
    def __init__(self, device="auto"):
        self.tokenizer = ModelLoader.get_tokenizer()
        self.model = ModelLoader.get_model(device=device)

    def evaluate(self, rubric, instruction, response):
        # Create a conversation template
        conv = get_conv_template("llama-2")
        conv.set_system_message("You are a fair evaluator language model.")

        # Add instruction and response
        conv.append_message(conv.roles[0], f"###Task Description: An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given. \
                                            1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general. \
                                            2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric. \
                                            3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\" \
                                            4. Please do not generate any other opening, closing, and explanations.")
        conv.append_message(conv.roles[0], f"###The instruction to evaluate: {instruction}")
        conv.append_message(conv.roles[0], f"###Response to evaluate: {response}")
        conv.append_message(conv.roles[0], f"###Score Rubric: {rubric}")
        conv.append_message(conv.roles[0], f"###Feedback")

        conv.append_message(conv.roles[1], None)  # Placeholder for the model's response

        # Generate the formatted prompt
        prompt = conv.get_prompt()

        # Tokenize the prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        # Generate model output
        outputs = self.model.generate(
            input_ids, do_sample=True, temperature=1.0, top_p=0.9,
            max_new_tokens=512, repetition_penalty=1.03
        )
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(decoded_output)

        # Extract feedback and score using regex
        match = re.search(r'Feedback:(.*?)\[\s*RESULT\s*\]\s*(\d+)', decoded_output, re.DOTALL)

        if match:
            return {"feedback": match.group(1).strip(), "score": int(match.group(2))}

        # If the first match fails, check for "overall score" followed by a number
        match_alt = re.search(r'So the overall score is.*?(\d+)', decoded_output, re.IGNORECASE)

        if match_alt:
            return {"feedback": "Extracted from 'overall score' reference", "score": int(match_alt.group(1))}

        return {"feedback": "Error parsing response", "score": None}

