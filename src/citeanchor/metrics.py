"""Detection metrics with fake citations (GT false) as the positive class, plus token cost."""
from __future__ import annotations

import collections


def metrics(TP: int, FN: int, FP: int, TN: int) -> dict:
    n = TP + FN + FP + TN
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    return {
        "N": n,
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "Acc": (TP + TN) / n if n else 0.0,
        "Prec": prec,
        "Rec": rec,
        "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "FPR": FP / (FP + TN) if FP + TN else 0.0,   # share of real citations wrongly flagged
    }


def confusion(rows) -> dict:
    """rows: iterable of (gt_is_real, pred_is_real). Fake is the positive class."""
    TP = FN = FP = TN = 0
    for gt_real, pred_real in rows:
        if not gt_real and not pred_real:
            TP += 1
        elif not gt_real and pred_real:
            FN += 1
        elif gt_real and not pred_real:
            FP += 1
        else:
            TN += 1
    return metrics(TP, FN, FP, TN)


# USD per 1M tokens (input, output); unlisted models get no cost fields.
PRICES = {
    "gpt-5.6-sol":   (5.00, 30.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna":  (0.20, 1.20),
    "gpt-4o":        (2.50, 10.00),
    "gemini-3-flash-preview": (0.50, 3.00),
}
SERPAPI_USD_PER_SEARCH = 0.01     # USD per SerpApi search

USAGE_KEYS = ("calls", "prompt_tokens", "completion_tokens", "reasoning_tokens")


def accumulate_usage(*snapshots) -> dict:
    """Sum token counters over several usage snapshots."""
    total = {k: 0 for k in USAGE_KEYS}
    for snap in snapshots:
        for k in USAGE_KEYS:
            total[k] += (snap or {}).get(k, 0) or 0
    return total


def cost_block(usage: dict, model: str) -> dict:
    """Token spend for a run, priced when the model's rate is known."""
    out = {"model": model, **usage}
    rate = PRICES.get(model)
    if rate:
        cin, cout = rate
        out["usd_input"] = round(usage["prompt_tokens"] / 1e6 * cin, 4)
        out["usd_output"] = round(usage["completion_tokens"] / 1e6 * cout, 4)
        out["usd_total"] = round(out["usd_input"] + out["usd_output"], 4)
    return out


def summarize(results: list) -> dict:
    """Full scorecard for a run, embedded in the output file's `summary`."""
    pairs = [(bool(r["GT"]), bool(r["Pred"])) for r in results]
    out = dict(confusion(pairs))
    out["correct"] = sum(1 for g, p in pairs if g == p)
    out["positive_class"] = "fake (GT=false)"
    methods = collections.Counter(
        r.get("verification", {}).get("method", "unknown") for r in results)
    out["by_method"] = dict(methods.most_common())
    return out
