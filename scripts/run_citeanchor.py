#!/usr/bin/env python3
"""Run CiteAnchor on a citation file; writes per-citation verdicts, traces and a summary.

    python3 scripts/run_citeanchor.py --file citations.json --output runs/run.json --no-timestamp --resume
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from citeanchor import runlog  # noqa: E402
from citeanchor.metrics import summarize
from citeanchor.paths import resolve_output
from citeanchor.connectors import SerpApi
from citeanchor.cascade import SearchCascade
from citeanchor.llm import model_name, usage as llm_usage
from citeanchor.parsing import build_query
from citeanchor.verdict import adjudicate

log = runlog.get("run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/citeaudit_realworld.json")
    ap.add_argument("--output", default="runs/citeanchor.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="number of citations (default: all)")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent citations; rate-limited connectors share one limiter, so more is not faster")
    ap.add_argument("--no-paid", action="store_true",
                    help="free tiers only — measures the cascade without SerpApi")
    ap.add_argument("--cache", default=None,
                    help="record cache file (default: the shared cache next to the package); "
                         "a run meant to measure retrieval should use a fresh file")
    ap.add_argument("--abort-on-degraded", action="store_true",
                    help="stop the run when the broad free indexes drop out of the fan-out "
                         "(see HEALTH_* below); partial results are still written")
    ap.add_argument("--skip", default="",
                    help="comma-separated free connectors to leave out of the fan-out, e.g. "
                         "'openalex'. Recorded in the log and the summary, so a connector that "
                         "was switched off can never be mistaken for one that failed")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <output>.partial.jsonl: rows already recorded "
                         "there are loaded and skipped. Needs --no-timestamp so the "
                         "output path, and so the checkpoint, is the same as last time")
    ap.add_argument("--no-timestamp", action="store_true",
                    help="write to --output exactly; without this a "
                         "timestamp is inserted so runs never overwrite each other")
    args = ap.parse_args()

    out_path = resolve_output(args.output, timestamp=not args.no_timestamp)
    log_path = runlog.setup(out_path.with_suffix(".log"))
    rows = json.loads(Path(args.file).read_text(encoding="utf-8"))["data"]
    rows = rows[args.start:] if args.limit is None else rows[args.start:args.start + args.limit]

    client = None
    if os.getenv("OPENAI_API_KEY"):
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    from citeanchor.cache import RecordCache
    casc = SearchCascade(use_paid=not args.no_paid, workers=6,
                         cache=RecordCache(args.cache) if args.cache else None)
    log.info("record cache: %s", args.cache or "shared default")
    skipped = sorted({x.strip().lower() for x in args.skip.split(",") if x.strip()})
    if skipped:
        unknown = [x for x in skipped if x not in {c.name for c in casc.free}]
        if unknown:
            raise SystemExit(f"--skip names no such free connector: {unknown}; "
                             f"known: {[c.name for c in casc.free]}")
        casc.free = [c for c in casc.free if c.name not in skipped]
        log.info("connectors skipped by request: %s", skipped)
    probe = casc.probe()
    log.info("connectors: %s", {k: ("up" if v else "down") for k, v in probe.items()})
    log.info("ACL mirror: %s", "ready" if casc.acl.ready else "not built")
    log.info("output: %s", out_path)
    log.info("log:    %s", log_path)
    log.info("%d citations, paid tier %s, backbone %s", len(rows), "OFF" if args.no_paid else "ON",
             model_name() if client else "none (regex parser, rules only)")
    log.debug("argv: %s", vars(args))

    results, lock, t0 = [], threading.Lock(), time.time()

    # finished rows are appended to a JSONL checkpoint as they complete
    ckpt_path = out_path.with_suffix(".partial.jsonl")
    if args.resume and ckpt_path.exists():
        # drop malformed lines (e.g. a truncated last line) and rewrite the file before appending
        done, dropped = [], 0
        for l in ckpt_path.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            try:
                done.append(json.loads(l))
            except json.JSONDecodeError:
                dropped += 1
        if dropped:
            log.warning("resume: %d malformed checkpoint line(s) dropped; those rows re-run", dropped)
        ckpt_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done),
                             encoding="utf-8")
        results.extend(done)
        seen = {r["id"] for r in done}
        log.info("resume: %d rows loaded from %s, %d of %d still to run",
                 len(done), ckpt_path, sum(1 for r in rows if r.get("id") not in seen), len(rows))
    else:
        seen = set()
        ckpt_path.write_text("", encoding="utf-8")
    ckpt = open(ckpt_path, "a", encoding="utf-8")

    # retrieval health: share of recent rows in which each free connector was queried
    HEALTH_EVERY, HEALTH_WINDOW = 200, 200
    HEALTH_FLOOR = {"crossref": 0.90, "semanticscholar": 0.70}
    fanouts: list = []          # one set of connector names per row that fanned out
    degraded = threading.Event()

    def participation(window: list) -> dict:
        names = sorted({n for f in window for n in f})
        return {n: round(sum(1 for f in window if n in f) / len(window), 3) for n in names}

    def one(offset: int, row: dict) -> dict:
        cid = row.get("id", offset + 1)
        runlog.set_item(f"id={cid}")
        started = time.time()
        log.debug("citation: %s", row["citation"])
        q = build_query(row["citation"], client)
        res = casc.search(q)
        v = adjudicate(res, client=client, casc=casc)
        gt = bool(row["GT"])
        log.debug("verdict: %s via rule=%s basis=%s conflicts=%s — %s",
                  "real" if v.is_real else "fake", v.rule, v.basis, v.conflicts,
                  v.reason)
        entry = {
            "index": offset,
            "id": cid,
            "source_type": row.get("source_type"),
            "citation": row["citation"],
            "GT": gt,
            "Pred": v.is_real,
            "Correct": v.is_real == gt,
            "verification": {
                # `method` is the retrieval tier that settled the row.
                "method": res.tier,
                "found": v.is_real,
                "basis": v.basis,
                "rule": v.rule,
                "conflicts": v.conflicts,
                "agreements": v.agreements,
                "reason": v.reason,
                "confidence": v.confidence,
                "connector_calls": res.connector_calls,
                "paid_calls": res.paid_calls,
                "evidence": v.evidence,
                # None when the judge was not called
                "judge": v.judge,
                "notes": res.notes,
                # proposed source record for a citation judged fake, or None
                "correction": getattr(v, "correction", None),
                # a recovered record that was not allowed to decide the verdict
                "recovered_evidence": getattr(v, "recovered_evidence", None),
                "recovery_gated": getattr(v, "recovery_gated", None),
            },
            # retrieval trace: query, connector calls, candidates
            "trace": {**res.to_dict(), "elapsed_s": round(time.time() - started, 2)},
        }
        with lock:
            results.append(entry)
            ckpt.write(json.dumps(entry, ensure_ascii=False) + "\n")
            ckpt.flush()
            n = len(results)
            free = {c.connector for c in res.calls if c.tier == "free"}
            if free:
                fanouts.append(free)
            if n % HEALTH_EVERY == 0 and len(fanouts) >= 50:
                part = participation(fanouts[-HEALTH_WINDOW:])
                low = {k: part.get(k, 0.0) for k, floor in HEALTH_FLOOR.items()
                       if part.get(k, 0.0) < floor}
                log.info("  health @%d: free-tier participation %s", n, part)
                if low:
                    log.warning("  RETRIEVAL DEGRADED: %s below their floors %s — verdicts in "
                                "this stretch rest on thinner evidence", low, HEALTH_FLOOR)
                    if args.abort_on_degraded:
                        degraded.set()
            log.info("  [%d/%d] id=%-4s GT=%-4s -> %-4s %s tier=%-12s %-18s %s (%.0fs)",
                     n, len(rows), cid, "real" if gt else "fake",
                     "real" if v.is_real else "fake",
                     "ok " if entry["Correct"] else "MISS",
                     res.tier, v.basis, v.conflicts, time.time() - t0)
        runlog.set_item(None)
        return entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        # checkpointed rows are skipped here so `index` stays the row's position in the input file
        futs = {ex.submit(one, args.start + i, r): (args.start + i, r)
                for i, r in enumerate(rows) if r.get("id") not in seen}
        for f in concurrent.futures.as_completed(futs):
            try:
                f.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as exc:
                log.exception("citation failed: %s: %s", type(exc).__name__, exc)
                # a failed row is recorded as wrong, with rule ERROR
                offset, row = futs[f]
                gt = bool(row["GT"])
                err = {"index": offset, "id": row.get("id", offset + 1),
                       "source_type": row.get("source_type"), "citation": row["citation"],
                       "GT": gt, "Pred": not gt, "Correct": False,
                       "error": f"{type(exc).__name__}: {exc}"[:300],
                       "verification": {"method": "error", "found": not gt, "basis": "error",
                                        "rule": "ERROR", "conflicts": [], "agreements": [],
                                        "reason": "pipeline raised; scored as wrong",
                                        "confidence": 0.0, "connector_calls": 0, "paid_calls": 0,
                                        "evidence": None, "judge": None, "notes": [],
                                        "correction": None, "recovered_evidence": None,
                                        "recovery_gated": None},
                       "trace": {"query": {}, "tier": "error", "calls": [], "candidates": []}}
                with lock:
                    results.append(err)
                    ckpt.write(json.dumps(err, ensure_ascii=False) + "\n")
                    ckpt.flush()
            if degraded.is_set():
                pending = [x for x in futs if x.cancel()]
                if pending:
                    log.error("aborting: retrieval degraded; %d citations not run, %d written",
                              len(pending), len(results))

    ckpt.close()
    results.sort(key=lambda r: r["index"])
    summary = summarize(results)
    summary["runtime"] = {"seconds": round(time.time() - t0, 1), "workers": args.workers}

    def tally(key, source=lambda r: r["verification"]):
        out = {}
        for r in results:
            out[source(r).get(key) or "-"] = out.get(source(r).get(key) or "-", 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    judged = [r for r in results if r["verification"].get("judge")]
    per_connector: dict = {}
    for r in results:
        for c in r["trace"]["calls"]:
            slot = per_connector.setdefault(
                c["connector"], {"calls": 0, "results": 0, "eligible": 0, "errors": 0})
            slot["calls"] += 1
            slot["results"] += c["n_results"]
            slot["eligible"] += c["n_eligible"]
            slot["errors"] += bool(c["error"])

    rec = [r for r in results if (r["trace"].get("recovery") or {})]
    summary["recovery"] = {
        "triggered": len(rec),
        "by_trigger": tally("trigger", lambda r: r["trace"].get("recovery") or {}),
        "settled_rung": tally("settled_rung", lambda r: r["trace"].get("recovery") or {}),
        "paid_calls": sum(r["trace"]["recovery"].get("paid_calls", 0) for r in rec),
        "flips": [{"id": r["id"], "GT": r["GT"],
                   "before": r["trace"]["recovery"].get("verdict_before"),
                   "after": r["trace"]["recovery"].get("verdict_after"),
                   "rule": r["verification"]["rule"]}
                  for r in rec
                  if r["trace"]["recovery"].get("verdict_before")
                  != r["trace"]["recovery"].get("verdict_after")],
        "gated": sum(1 for r in rec if r["trace"]["recovery"].get("gated")),
    }
    summary["cascade"] = {
        "free_tier_participation": participation(fanouts) if fanouts else {},
        "rows_reaching_free_tier": len(fanouts),
        # share of rows answered from the record cache
        "cache_settled_share": (round(sum(1 for r in results if r["trace"].get("tier") == "cache")
                                      / len(results), 3) if results else 0.0),
        "connector_trips": {c.name: getattr(type(c), "trips", 0)
                            for c in casc.free if getattr(type(c), "trips", 0)},
        "aborted_degraded": degraded.is_set(),
        "skipped_connectors": skipped,
        # set when the SerpApi quota ran out during the run
        "serpapi_exhausted": bool(getattr(SerpApi, "exhausted", False)),
        "serpapi_exhausted_after_calls": getattr(SerpApi, "exhausted_after", None),
        "paid_calls_total": sum(r["verification"]["paid_calls"] for r in results),
        "connector_calls_total": sum(r["verification"]["connector_calls"] for r in results),
        "by_basis": {b: sum(1 for r in results if r["verification"]["basis"] == b)
                     for b in ("evidence_match", "evidence_conflict", "no_evidence")},
        "by_rule": tally("rule"),
        "by_connector": per_connector,
        "cache": casc.cache.stats(),
    }
    summary["llm"] = {**llm_usage(), "model": model_name()}
    summary["judge"] = {
        "invoked": len(judged),
        "by_trigger": {t: sum(1 for r in judged
                              if r["verification"]["judge"].get("_trigger") == t)
                       for t in sorted({r["verification"]["judge"].get("_trigger")
                                        for r in judged})},
        "by_verdict": {t: sum(1 for r in judged
                              if r["verification"]["judge"].get("verdict") == t)
                       for t in sorted({r["verification"]["judge"].get("verdict")
                                        for r in judged})},
    }

    out = {"source_file": str(Path(args.file).resolve()),
           "system": "citeanchor", "paid_tier": not args.no_paid,
           "log_file": str(log_path) if log_path else None,
           "results": results, "summary": summary}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("=" * 62)
    log.info("  detection  Acc=%.3f  Prec=%.3f  Rec=%.3f  F1=%.3f",
             summary["Acc"], summary["Prec"], summary["Rec"], summary["F1"])
    log.info("  retrieval  free-tier participation=%s  cache-settled=%.0f%%  trips=%s",
             summary["cascade"]["free_tier_participation"],
             100 * summary["cascade"]["cache_settled_share"],
             summary["cascade"]["connector_trips"] or "none")
    log.info("  paid calls: %s   free calls: %s",
             summary["cascade"]["paid_calls_total"],
             summary["cascade"]["connector_calls_total"])
    log.info("  verdict basis: %s", summary["cascade"]["by_basis"])
    log.info("  by rule:       %s", summary["cascade"]["by_rule"])
    log.info("  judge:         %s", summary["judge"])
    log.info("=" * 62)
    log.info("wrote %s", out_path)
    log.info("log   %s", log_path)


if __name__ == "__main__":
    main()
