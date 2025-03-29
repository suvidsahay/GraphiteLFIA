import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelLoader:
    _model = None
    _tokenizer = None

    @classmethod
    def get_model(cls, model_name="kaist-ai/Prometheus-7b-v1.0", device="auto"):
        if cls._model is None:
            print("Loading model...")
            with accelerator.main_process_first():
                # LLM setup
                cls._model = LlamaForCausalLM.from_pretrained(model_name, device_map=device)

        return cls._model

    @classmethod
    def get_tokenizer(cls, tokenizer_name="meta-llama/Llama-2-7b-chat-hf"):
        if cls._tokenizer is None:
            print("Loading tokenizer...")
            with accelerator.main_process_first():
                cls._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return cls._tokenizer

accelerator = Accelerator()