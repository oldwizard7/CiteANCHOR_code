#!/usr/bin/env python3
"""Print detection metrics for one or more run files (positive class: GT == false).

    python3 scripts/score_detection.py runs/run.json [runs/other.json ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from citeanchor.metrics import PRICES, SERPAPI_USD_PER_SEARCH, confusion  # noqa: E402


def searches(run: dict) -> int:
    s = run.get("summary") or {}
    if (s.get("cascade") or {}).get("paid_calls_total") is not None:
        return s["cascade"]["paid_calls_total"]
    return sum((r.get("verification") or {}).get("paid_calls", 0) or 0 for r in run["results"])


def main(paths):
    print(f"{'run':40s} {'N':>5} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>5}  {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6}")
    for p in paths:
        run = json.load(open(p))
        R = [r for r in run["results"]]
        m = confusion([(bool(r["GT"]), bool(r["Pred"])) for r in R])
        errors = sum(1 for r in R if (r.get("verification") or {}).get("rule") == "ERROR")
        print(f"{Path(p).name[:40]:40s} {m['N']:5d} {m['TP']:4d} {m['FN']:4d} {m['FP']:4d} {m['TN']:5d}  "
              f"{m['Acc']:.4f} {m['Prec']:.4f} {m['Rec']:.4f} {m['F1']:.4f} {m['FPR']:.4f}")
        s = run.get("summary") or {}
        llm = s.get("llm") or s.get("cost") or {}
        n_search = searches(run)
        model = llm.get("model", "?")
        rate = PRICES.get(model)
        tok = ""
        if llm.get("prompt_tokens") is not None:
            tok = f"tokens in {llm.get('prompt_tokens', 0):,} out {llm.get('completion_tokens', 0):,} ({model})"
            if rate:
                usd = llm.get("prompt_tokens", 0) / 1e6 * rate[0] + llm.get("completion_tokens", 0) / 1e6 * rate[1]
                tok += f" ≈ ${usd:.2f}"
        print(f"{'':40s} searches {n_search:,} (≈ ${n_search * SERPAPI_USD_PER_SEARCH:.2f}) | {tok}"
              + (f" | pipeline errors scored as wrong: {errors}" if errors else ""))
        rules = (s.get("cascade") or {}).get("by_rule")
        if rules:
            top = ", ".join(f"{k} {v}" for k, v in list(rules.items())[:8])
            print(f"{'':40s} deciding rules: {top}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run file(s) written by the runners")
    main(ap.parse_args().runs)
