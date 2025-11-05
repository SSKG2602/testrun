import csv
from typing import Optional

from autogen import AssistantAgent, UserProxyAgent

from cost_model import compute_cost
from local_llm import LocalLLM


def run_autogen_like(path_csv: str, model_name: Optional[str] = None):
    backend = LocalLLM(model_name)
    state = {"meta": None, "label": None}

    def completion_fn(messages, **kwargs):
        label, meta = backend.classify_email(state["subject"], state["body"])
        state["meta"] = meta
        state["label"] = label
        return {"role": "assistant", "content": label}

    assistant = AssistantAgent(
        name="assistant",
        llm_config={"config_list": [{"model": "local-llm", "completion_fn": completion_fn}]},
    )
    user = UserProxyAgent(name="user", human_input_mode="NEVER")

    total_cost = 0.0
    total_ms = 0
    total_tokens = 0

    with open(path_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        state["subject"] = row["subject"]
        state["body"] = row["body"]
        state["meta"] = None
        state["label"] = None

        user.initiate_chat(
            assistant,
            message=(
                "Classify the email into one of ['spam','billing','urgent','other'] and respond with only the label.\n\n"
                f"Subject: {row['subject']}\nBody:\n{row['body']}"
            ),
        )

        meta = state["meta"] or {"tokens_in": 0, "tokens_out": 0, "seconds": 0.0}
        cost = compute_cost(meta["tokens_in"], meta["tokens_out"], meta["seconds"])
        total_cost += cost
        total_ms += int(meta["seconds"] * 1000)
        total_tokens += meta["tokens_in"] + meta["tokens_out"]

        assistant.reset()
        user.reset()

    return {
        "Agent": "AutoGen",
        "Run_Type": "N/A",
        "Total_Cost_USD": round(total_cost, 6),
        "Total_Latency_ms": total_ms,
        "Total_LLM_Tokens": total_tokens,
        "Suppression_Rate_Percent": 0.0,
    }
