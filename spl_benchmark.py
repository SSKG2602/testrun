#!/usr/bin/env python3
"""
Dataset utilities and SPL orchestration helpers.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from spl_test import run_spl

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "test_emails.csv"
LEARNED_PATH = ROOT / "learned_patterns.json"
CHART_PATH = ROOT / "benchmark_chart.png"

RANDOM_SEED = 2602
DISTRIBUTION = {"spam": 30, "billing": 30, "urgent": 20, "other": 20}


def make_dataset(path: Path = DATA_PATH) -> None:
    random.seed(RANDOM_SEED)

    rows: List[Dict[str, str]] = []

    def emit(label: str, sender: str, subject: str, body: str) -> None:
        rows.append({"sender": sender, "subject": subject, "body": body, "label": label})

    for _ in range(DISTRIBUTION["spam"]):
        emit(
            "spam",
            "spam@example.com",
            random.choice(["WIN BIG NOW!!!", "Exclusive Offer Inside", "You won a prize", "Free gift card"]),
            "Limited time! Click now for rewards and bonuses.",
        )
    for _ in range(DISTRIBUTION["billing"]):
        emit(
            "billing",
            "billing@vendor.com",
            random.choice(["Invoice available", "Your payment receipt", "Billing notice", "Payment due"]),
            f"Invoice #{random.randint(1000, 9999)} attached. Please review your payment.",
        )
    for _ in range(DISTRIBUTION["urgent"]):
        emit(
            "urgent",
            "boss@acme.com",
            random.choice(["URGENT: Meeting moved", "ASAP: Budget review", "Urgent: Action required", "ASAP: pricing"]),
            "Please respond immediately. This is time-sensitive.",
        )
    for _ in range(DISTRIBUTION["other"]):
        emit(
            "other",
            "friend@gmail.com",
            random.choice(["Coffee tomorrow?", "Weekend plans", "Photos from trip", "Team lunch next week"]),
            "Hey! Thought you might like these updates.",
        )

    random.shuffle(rows)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sender", "subject", "body", "label"])
        writer.writeheader()
        writer.writerows(rows)


def run_spl_cold_then_warm(dataset_path: str, model_name: Optional[str] = None) -> List[Dict[str, object]]:
    if LEARNED_PATH.exists():
        LEARNED_PATH.unlink()
    cold = run_spl(dataset_path, warm_start=False, model_name=model_name)
    warm = run_spl(dataset_path, warm_start=True, model_name=model_name)
    return [cold, warm]


def write_chart(rows: List[Dict[str, object]], out_path: Path = CHART_PATH) -> None:
    df = pd.DataFrame(rows)
    subset = df[(df["Agent"] != "SPL") | (df["Run_Type"] == "Warm_Start")]
    order = ["LangChain", "AutoGen", "SPL"]
    costs = []
    labels = []
    for agent in order:
        run_type = "Warm_Start" if agent == "SPL" else "N/A"
        selection = subset[(subset["Agent"] == agent) & (subset["Run_Type"] == run_type)]
        if selection.empty:
            continue
        cost = float(selection.iloc[0]["Total_Cost_USD"]) * 10.0
        costs.append(cost)
        labels.append(agent)

    if not costs:
        return

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, costs, color=["#5B8FF9", "#5AD8A6", "#FF6F6F"])
    plt.title("Benchmark: Agent Cost per 1000 Emails")
    plt.ylabel("Total Cost (USD)")
    for bar, value in zip(bars, costs):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value, f"${value:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset utilities for SPL benchmarks.")
    parser.add_argument("--setup", action="store_true", help="Create the benchmark dataset.")
    parser.add_argument("--dataset", default=str(DATA_PATH))
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-spl", action="store_true", help="Run SPL cold & warm and print metrics.")
    args = parser.parse_args()

    if args.setup or not Path(args.dataset).exists():
        make_dataset(Path(args.dataset))
        print(f"Created {args.dataset}")

    if args.run_spl:
        rows = run_spl_cold_then_warm(args.dataset, args.model)
        print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
