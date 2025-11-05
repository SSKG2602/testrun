#!/usr/bin/env python3
"""
SPL agent stack powered by a local Hugging Face model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cost_model import compute_cost
from local_llm import LocalLLM

ROOT = Path(__file__).parent
LEARNED_PATH = ROOT / "learned_patterns.json"

BLACKLIST = {"spam@example.com"}
MAX_BODY_LENGTH = 20_000
MIN_CONFIDENCE = 0.85

_LLM_SINGLETON: Optional[LocalLLM] = None
_LLM_MODEL_NAME: Optional[str] = None


def _llm(model_name: Optional[str] = None) -> LocalLLM:
    global _LLM_SINGLETON, _LLM_MODEL_NAME
    target = model_name or os.getenv("HF_MODEL")
    if _LLM_SINGLETON is None or _LLM_MODEL_NAME != target:
        _LLM_SINGLETON = LocalLLM(model_name)
        _LLM_MODEL_NAME = target
    return _LLM_SINGLETON


def _load_patterns() -> Dict[str, Dict[str, Any]]:
    if not LEARNED_PATH.exists():
        return {}
    with LEARNED_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_patterns(patterns: Dict[str, Dict[str, Any]]) -> None:
    with LEARNED_PATH.open("w", encoding="utf-8") as handle:
        json.dump(patterns, handle, indent=2)


def _match_pattern(subject_lower: str, patterns: Dict[str, Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    for name, pattern in patterns.items():
        confidence = float(pattern.get("confidence", 0.0))
        if confidence < MIN_CONFIDENCE:
            continue
        keywords = [kw.lower() for kw in pattern.get("rules", {}).get("subject_contains", [])]
        if any(keyword in subject_lower for keyword in keywords):
            return pattern.get("result", "other"), name
    return None


def _learn_pattern(label: str, patterns: Dict[str, Dict[str, Any]]) -> None:
    if label == "urgent":
        patterns["urgent"] = {
            "name": "urgent",
            "rules": {"subject_contains": ["urgent", "asap"]},
            "confidence": 0.9,
            "result": "urgent",
        }
    elif label == "billing":
        patterns["billing"] = {
            "name": "billing",
            "rules": {"subject_contains": ["invoice", "payment", "receipt", "bill"]},
            "confidence": 0.9,
            "result": "billing",
        }


def run_spl(dataset_path: str, warm_start: bool = False, model_name: Optional[str] = None) -> Dict[str, Any]:
    with open(dataset_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not warm_start and LEARNED_PATH.exists():
        LEARNED_PATH.unlink()

    patterns = _load_patterns() if warm_start else {}

    totals = {
        "total_cost_usd": 0.0,
        "total_latency_ms": 0,
        "total_tokens": 0,
        "avoided_l2": 0,
    }

    for email in rows:
        subject = email.get("subject", "") or ""
        subject_lower = subject.lower()
        body = email.get("body", "") or ""
        sender = email.get("sender", "") or ""

        method = "l2_llm"

        if len(body) > MAX_BODY_LENGTH:
            method = "l0_max_length"
            label = "other"
            cost = 0.0
            tokens = 0
            latency_ms = 0
        elif sender in BLACKLIST:
            method = "l0_blacklist"
            label = "spam"
            cost = 0.0
            tokens = 0
            latency_ms = 0
        else:
            matched = _match_pattern(subject_lower, patterns)
            if matched:
                label, pattern_name = matched
                method = f"l1_pattern:{pattern_name}"
                cost = 0.0
                tokens = 0
                latency_ms = 0
            else:
                label, meta = _llm(model_name).classify_email(subject, body)
                cost = compute_cost(meta["tokens_in"], meta["tokens_out"], meta["seconds"])
                tokens = meta["tokens_in"] + meta["tokens_out"]
                latency_ms = int(meta["seconds"] * 1000)
                _learn_pattern(label, patterns)

        if method != "l2_llm":
            totals["avoided_l2"] += 1

        totals["total_cost_usd"] += cost
        totals["total_latency_ms"] += latency_ms
        totals["total_tokens"] += tokens

    _save_patterns(patterns)

    suppression_rate = round(100.0 * totals["avoided_l2"] / max(1, len(rows)), 2)

    return {
        "Agent": "SPL",
        "Run_Type": "Warm_Start" if warm_start else "Cold_Start",
        "Total_Cost_USD": round(totals["total_cost_usd"], 6),
        "Total_Latency_ms": totals["total_latency_ms"],
        "Total_LLM_Tokens": totals["total_tokens"],
        "Suppression_Rate_Percent": suppression_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SPL agent on a dataset.")
    parser.add_argument("--dataset", default=str(ROOT / "test_emails.csv"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", choices=["cold", "warm"], default="cold")
    args = parser.parse_args()

    metrics = run_spl(args.dataset, warm_start=args.mode == "warm", model_name=args.model)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
