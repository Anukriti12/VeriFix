"""Print the headline, go/no-go, per-class and per-k tables from results/.
Usage: python -m scripts.aggregate_results --results results/"""
import argparse
from verifix.eval.aggregate import report

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/")
    report(ap.parse_args().results)
