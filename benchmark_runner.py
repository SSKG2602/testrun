import csv
import argparse
from pathlib import Path

from autogen_test import run_autogen_like
from langchain_test import run_langchain_like
from spl_benchmark import write_chart, run_spl_cold_then_warm, make_dataset

FIELDS = [
    "Agent",
    "Run_Type",
    "Total_Cost_USD",
    "Total_Latency_ms",
    "Total_LLM_Tokens",
    "Suppression_Rate_Percent",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="test_emails.csv")
    ap.add_argument("--model", default=None, help="HF model name (env HF_MODEL otherwise)")
    ap.add_argument("--out_csv", default="benchmark_results.csv")
    ap.add_argument("--chart", default="benchmark_chart.png")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        make_dataset(dataset_path)

    rows = []
    rows.append(run_langchain_like(str(dataset_path), args.model))
    rows.append(run_autogen_like(str(dataset_path), args.model))
    rows.extend(run_spl_cold_then_warm(str(dataset_path), args.model))

    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out_csv}")

    write_chart(rows, Path(args.chart))
    print(f"Wrote {args.chart}")


if __name__ == "__main__":
    main()
