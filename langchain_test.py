import csv
from typing import Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM

from local_llm import LocalLLM
from cost_model import compute_cost


class LangChainLocalLLM(LLM):
    def __init__(self, model_name: Optional[str] = None):
        super().__init__()
        self.backend = LocalLLM(model_name)
        self.last_meta = None

    @property
    def _llm_type(self) -> str:
        return "local_llm"

    def _call(self, prompt: str, stop: Optional[list[str]] = None, run_manager=None) -> str:
        text, t_in, t_out, secs = self.backend.generate(prompt)
        if stop:
            for token in stop:
                if token in text:
                    text = text.split(token)[0]
                    break
        self.last_meta = {"tokens_in": t_in, "tokens_out": t_out, "seconds": secs}
        return text


def _normalize_label(raw: str) -> str:
    lower = raw.strip().lower()
    if "spam" in lower:
        return "spam"
    if "urgent" in lower:
        return "urgent"
    if any(k in lower for k in ("bill", "invoice", "payment", "receipt")):
        return "billing"
    return "other"


def run_langchain_like(path_csv: str, model_name: Optional[str] = None):
    prompt = PromptTemplate.from_template(
        "You are a strict classifier. Return exactly one label from this set: ['spam','billing','urgent','other']\n\n"
        "Subject: {subject}\nBody:\n{body}\n\nLabel:"
    )
    llm = LangChainLocalLLM(model_name)
    chain = LLMChain(prompt=prompt, llm=llm)

    total_cost = 0.0
    total_ms = 0
    total_tokens = 0

    with open(path_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        output = chain.invoke({"subject": row["subject"], "body": row["body"]})
        text = output["text"] if isinstance(output, dict) else output
        label = _normalize_label(text)
        _ = label
        meta = llm.last_meta or {"tokens_in": 0, "tokens_out": 0, "seconds": 0.0}
        cost = compute_cost(meta["tokens_in"], meta["tokens_out"], meta["seconds"])
        total_cost += cost
        total_ms += int(meta["seconds"] * 1000)
        total_tokens += meta["tokens_in"] + meta["tokens_out"]

    return {
        "Agent": "LangChain",
        "Run_Type": "N/A",
        "Total_Cost_USD": round(total_cost, 6),
        "Total_Latency_ms": total_ms,
        "Total_LLM_Tokens": total_tokens,
        "Suppression_Rate_Percent": 0.0,
    }
