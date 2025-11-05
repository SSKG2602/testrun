import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
FALLBACK_MODEL = os.getenv("HF_FALLBACK_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def _load_model(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    return tokenizer, model


class LocalLLM:
    def __init__(self, model_name: str = None, max_new_tokens: int = 64):
        name = model_name or HF_MODEL
        try:
            self.tokenizer, self.model = _load_model(name)
        except Exception:
            self.tokenizer, self.model = _load_model(FALLBACK_MODEL)
        self.max_new_tokens = max_new_tokens

    def _count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def generate(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        t_in = inputs["input_ids"].shape[-1]
        start = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )
        secs = time.perf_counter() - start
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        gen = text[len(prompt):] if text.startswith(prompt) else text
        t_out = self._count(gen)
        return gen.strip(), t_in, t_out, secs

    def classify_email(self, subject: str, body: str):
        prompt = (
            "You are a strict classifier. Return exactly one label from this set: "
            "['spam','billing','urgent','other']\n\n"
            f"Subject: {subject}\nBody:\n{body}\n\nLabel:"
        )
        text, t_in, t_out, secs = self.generate(prompt)
        raw = text.lower().strip()
        if "spam" in raw:
            label = "spam"
        elif "urgent" in raw:
            label = "urgent"
        elif any(k in raw for k in ["bill", "invoice", "payment", "receipt"]):
            label = "billing"
        elif raw in {"other", "none", "misc"}:
            label = "other"
        else:
            label = "other"
        return label, {"tokens_in": t_in, "tokens_out": t_out, "seconds": secs}
