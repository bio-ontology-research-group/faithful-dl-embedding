#!/usr/bin/env python3
"""Evaluate instance-retrieval predictions against reasoner ground truth.

Input predictions map query IDs to individual IRIs. Outputs per-KB/query-type
Jaccard, precision, recall, F1 and TBox-only recall as JSON and Markdown.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from common import DATA_ROOT, NAMES


def scores(predicted, truth):
    """Return Jaccard and set precision/recall/F1 with empty-set conventions."""
    p,t=set(predicted),set(truth); inter=len(p&t)
    j=1.0 if not p and not t else inter/len(p|t)
    precision=1.0 if not p and not t else (inter/len(p) if p else 0.0)
    recall=1.0 if not p and not t else (inter/len(t) if t else 0.0)
    f1=1.0 if precision+recall==0 and not p and not t else (2*precision*recall/(precision+recall) if precision+recall else 0.0)
    return {"jaccard":j,"precision":precision,"recall":recall,"f1":f1}


def evaluate(predictions: dict, ground_truth: dict, target: str="full") -> dict:
    """Compute mean metrics grouped by query type, including TBox-needed recall."""
    if target not in ("full","el"): raise ValueError("target must be full or el")
    groups=defaultdict(list)
    for q in ground_truth["queries"]:
        truth=q.get(f"{target}_answers")
        if truth is None: continue
        pred=predictions.get(q["query_id"],[]); row=scores(pred,truth)
        needed=set(truth)-set(q.get("abox_lookup",[]))
        row["tbox_recall"]=(len(set(pred)&needed)/len(needed)) if needed else None
        row["n_tbox_answers"]=len(needed);row["n_answers"]=len(truth);row["type"]=q["type"]
        groups[q["type"]].append(row)
    result={}
    for typ,rows in sorted(groups.items()):
        result[typ]={k:sum(r[k] for r in rows)/len(rows) for k in ("jaccard","precision","recall","f1")}
        vals=[r["tbox_recall"] for r in rows if r["tbox_recall"] is not None]
        result[typ]["tbox_recall"]=sum(vals)/len(vals) if vals else None
        result[typ]["share_answers_needing_tbox"]=sum(r["n_tbox_answers"] for r in rows)/sum(r["n_answers"] for r in rows) if sum(r["n_answers"] for r in rows) else 0.0
        result[typ]["n_queries"]=len(rows)
    return {"kb":ground_truth["kb"],"target":target,"metrics_by_type":result}


def _md_table(results, metric_keys):
    lines=["| KB / type | "+" | ".join(metric_keys)+" |","|---|"+"|".join(["---:"]*len(metric_keys))+"|"]
    for kb,types in results.items():
        for typ,vals in types.items():
            cells=["—" if vals.get(k) is None else f"{vals[k]:.3f}" for k in metric_keys]
            lines.append(f"| {kb} / {typ} | "+" | ".join(cells)+" |")
    return "\n".join(lines)



TYPE_ORDER = ("A", "A_and_B", "some_r_A", "A_and_some_r_B", "some_r_and_B", "some_r_some_s_A", "some_inv_r_A")


def _types(gt: dict) -> list[str]:
    present = {q["type"] for q in gt["queries"]}
    return [t for t in TYPE_ORDER if t in present] + sorted(present - set(TYPE_ORDER))


def _fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _missed_share(rows: list[dict], target: str, predicted: dict | None, field: str | None = None) -> float | None:
    """Answer-weighted share of ``target`` reference answers absent from a lookup.

    ``field`` names a per-query list in the ground truth (``abox_lookup``); otherwise
    ``predicted`` maps query ids to lists (told-hierarchy predictions).
    """
    rows = [q for q in rows if q.get(f"{target}_answers") is not None]
    if field is None and predicted is None:
        return None
    total = sum(len(q[f"{target}_answers"]) for q in rows)
    if not total:
        return None
    missed = 0
    for q in rows:
        found = set(q.get(field) or []) if field else set(predicted.get(q["query_id"], []))
        missed += len(set(q[f"{target}_answers"]) - found)
    return missed / total


def kb_type_summary(gt: dict, told: dict | None) -> dict[str, dict]:
    """Per query type: counts, TBox-needed / told-missed shares and full-vs-EL agreement.

    Parameters
    ----------
    gt : dict
        ``ground_truth.json`` content (C1 or C2 format).
    told : dict or None
        Told-hierarchy predictions (query id to answers), if available.

    Returns
    -------
    dict
        Type to metrics; shares are answer-weighted, ``None`` when undefined.
    """
    out = {}
    for typ in _types(gt):
        rows = [q for q in gt["queries"] if q["type"] == typ]
        both = [q for q in rows if q.get("full_answers") is not None and q.get("el_answers") is not None]
        full_total = sum(len(q["full_answers"]) for q in both)
        out[typ] = {
            "queries": len(rows),
            "el_refs": sum(q.get("el_answers") is not None for q in rows),
            "full_refs": sum(q.get("full_answers") is not None for q in rows),
            "full_timeouts": sum(bool(q.get("full_timeout")) for q in rows),
            "needs_tbox_el": _missed_share(rows, "el", None, "abox_lookup"),
            "told_missed_el": _missed_share(rows, "el", told),
            "needs_tbox_full": _missed_share(rows, "full", None, "abox_lookup"),
            "told_missed_full": _missed_share(rows, "full", told),
            "agree_queries": len(both),
            "agree_exact": sum(set(q["full_answers"]) == set(q["el_answers"]) for q in both),
            "agree_mean_jaccard": (sum(scores(q["el_answers"], q["full_answers"])["jaccard"] for q in both) / len(both)) if both else None,
            "full_loss_share": (sum(len(set(q["full_answers"]) - set(q["el_answers"])) for q in both) / full_total) if full_total else None,
        }
        checks = [q["elk_vs_hermit"] for q in rows if q.get("elk_vs_hermit") is not None]
        out[typ]["elk_check"] = None if not checks else {
            "compared": len(checks), "disagree": sum(not x["equal"] for x in checks),
            "missing_answers": sum(x["missing_in_elk"] for x in checks), "extra_answers": sum(x["extra_in_elk"] for x in checks),
            "hermit_answers": sum(len(q["el_hermit_answers"]) for q in rows if q.get("elk_vs_hermit") is not None)}
        for target in ("el", "full"):
            cmps = [q.get(f"filter_vs_{target}") for q in rows if q.get(f"filter_vs_{target}") is not None]
            out[typ][f"filter_{target}"] = None if not cmps else {
                "compared": len(cmps), "disagree": sum(not c["equal"] for c in cmps),
                "missing_queries": sum(c["missing_from_filter"] > 0 for c in cmps),
                "missing_answers": sum(c["missing_from_filter"] for c in cmps),
                "extra_queries": sum(c["extra_in_filter"] > 0 for c in cmps),
                "extra_answers": sum(c["extra_in_filter"] for c in cmps)}
    return out


def build_retrieval_report(root: Path) -> str:
    """Assemble the multi-KB Markdown report from ``ground_truth.json`` and lookup prediction files."""
    from common import read_json
    loaded, summaries, tables = {}, {}, {}
    for kb in NAMES:
        d = root / "kbs" / kb
        gt = read_json(d / "ground_truth.json")
        if gt is None:
            continue
        loaded[kb] = gt
        told = read_json(d / "predictions_told.json")
        summaries[kb] = kb_type_summary(gt, told)
        tables[kb] = {}
        for target in ("full", "el"):
            for baseline, file in (("ABox lookup", "predictions_abox.json"), ("Told hierarchy", "predictions_told.json")):
                pred = read_json(d / file)
                if pred is not None:
                    tables[kb][f"{target}/{baseline}"] = evaluate(pred, gt, target)["metrics_by_type"]
    manifest = read_json(root / "kbs" / "manifest.json", {}) or {}
    c = ["# Instance-retrieval benchmark results", "",
         "Generated by `evaluate.py --report` from each KB's `ground_truth.json`, `predictions_abox.json` "
         "(plain asserted-ABox lookup) and `predictions_told.json` (ABox lookup after closing class assertions under "
         "the told named-class hierarchy). Shares are answer-weighted over the queries of a KB and type that have the "
         "reference in question. Jaccard uses 1 when both sets are empty.", "",
         "* **EL reference** (`el_answers`): ELK 0.4.3 on the EL projection for all non-inverse types; HermiT 1.3.8 on the "
         "EL projection for `some_inv_r_A` (C2 pipeline). Family and Father keep their C1 ground truth (HermiT on the "
         "projection for every type).",
         "* **Full-KB reference** (`full_answers`): HermiT on the full KB, per-query watchdog timeout and a global budget "
         "per KB; queries without it are excluded from every full-KB column.",
         "* **needs TBox**: share of reference answers not returned by plain ABox lookup. **told-missed**: share still not "
         "returned by the told-hierarchy lookup.",
         "* **full vs EL**: over queries with both references; *exact* counts identical answer sets, *loss* is the share of "
         "full-KB answers absent from the EL reference.", ""]
    c += ["## Coverage", "", "| KB | sampler | queries | EL refs | full-KB refs | full timeouts | full-KB status | per-query timeout (s) | full budget (min) | ground-truth time (s) |",
          "|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for kb, gt in loaded.items():
        rows = summaries[kb].values()
        c.append(f"| {kb} | {gt.get('sampler') or 'C1 (reasoner per candidate)'} | {sum(r['queries'] for r in rows)} | "
                 f"{sum(r['el_refs'] for r in rows)} | {sum(r['full_refs'] for r in rows)} | {sum(r['full_timeouts'] for r in rows)} | "
                 f"{gt.get('full_status')} | {_fmt(gt.get('timeout_seconds_per_query'), 0)} | {_fmt(gt.get('budget_minutes_full'), 0)} | "
                 f"{_fmt(gt.get('seconds_total'), 0)} |")
    missing = [kb for kb in NAMES if kb not in loaded]
    if missing:
        c += ["", "No ground truth for: " + ", ".join(
            f"{kb} ({manifest.get('datasets', {}).get(kb, {}).get('status', 'not fetched')})" for kb in missing) + "."]
    c += ["", "## Reference answers by KB and query type", "",
          "| KB | type | queries | full-KB refs | needs TBox (EL) | told-missed (EL) | needs TBox (full) | told-missed (full) | full vs EL: queries | exact | mean Jaccard | loss |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kb, types in summaries.items():
        for typ, r in types.items():
            c.append(f"| {kb} | {typ} | {r['queries']} | {r['full_refs']} | {_fmt(r['needs_tbox_el'])} | {_fmt(r['told_missed_el'])} | "
                     f"{_fmt(r['needs_tbox_full'])} | {_fmt(r['told_missed_full'])} | {r['agree_queries']} | {r['agree_exact']} | "
                     f"{_fmt(r['agree_mean_jaccard'])} | {_fmt(r['full_loss_share'])} |")
    c += ["", "## Sampling filter vs reasoner", "",
          "C2 samples queries with a one-pass materialisation (`materialise.py`): ELK class realisation of the EL projection "
          "plus the full KB's asserted role edges closed under sub-/equivalent, inverse and symmetric properties and transitivity. "
          "*missing* = queries (answers) where the reasoner returns individuals the filter lacks, i.e. where the "
          "materialisation is incomplete (typically existential knowledge in the TBox); *extra* = the filter returns "
          "individuals the reasoner does not, e.g. edges from inverse/symmetric axioms that the EL projection drops.", "",
          "| KB | type | vs EL: compared | disagree | missing q (ans) | extra q (ans) | vs full: compared | disagree | missing q (ans) | extra q (ans) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    any_filter = False
    for kb, types in summaries.items():
        for typ, r in types.items():
            el, full = r["filter_el"], r["filter_full"]
            if el is None and full is None:
                continue
            any_filter = True
            cell = lambda f: ("— | — | — | —" if f is None else  # noqa: E731
                              f"{f['compared']} | {f['disagree']} | {f['missing_queries']} ({f['missing_answers']}) | {f['extra_queries']} ({f['extra_answers']})")
            c.append(f"| {kb} | {typ} | {cell(el)} | {cell(full)} |")
    if not any_filter:
        c.append("| — | — | — | — | — | — | — | — | — | — |")
    c += ["", "## ELK vs HermiT on the EL projection", "",
          "ELK 0.4.3 ignores `ObjectPropertyRange` and `DifferentIndividuals` axioms, which the EL projection keeps. "
          "For non-inverse queries, HermiT also answered on the projection; *missing* = answers HermiT finds and ELK (the EL "
          "reference) does not. Rows are shown only where the check ran.", "",
          "| KB | type | compared | disagree | missing answers (share of HermiT answers) | extra answers |", "|---|---|---:|---:|---:|---:|"]
    any_check = False
    for kb, types in summaries.items():
        for typ, r in types.items():
            x = r.get("elk_check")
            if x is None:
                continue
            any_check = True
            share = x["missing_answers"] / x["hermit_answers"] if x["hermit_answers"] else None
            c.append(f"| {kb} | {typ} | {x['compared']} | {x['disagree']} | {x['missing_answers']} ({_fmt(share)}) | {x['extra_answers']} |")
    if not any_check:
        c.append("| — | — | — | — | — | — |")
    for target in ("full", "el"):
        c += ["", f"## Baselines against {'full-KB' if target == 'full' else 'EL'} references", ""]
        for baseline in ("ABox lookup", "Told hierarchy"):
            flat = {kb: tables[kb][f"{target}/{baseline}"] for kb in tables if tables[kb].get(f"{target}/{baseline}")}
            c += [f"### {baseline}", "", _md_table(flat, ["jaccard", "precision", "recall", "f1", "tbox_recall"]), ""]
    return "\n".join(c) + "\n"


def main():
    """Run one prediction evaluation or regenerate the multi-KB report."""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--kb",choices=NAMES);p.add_argument("--data-dir",type=Path,default=DATA_ROOT);p.add_argument("--predictions",type=Path);p.add_argument("--target",choices=("full","el"),default="full");p.add_argument("--output",type=Path);p.add_argument("--report",action="store_true");a=p.parse_args()
    if a.report:
        report=build_retrieval_report(a.data_dir);out=a.output or Path(__file__).with_name("RETRIEVAL.md");out.write_text(report);print(f"wrote {out}")
    if a.kb:
        d=a.data_dir/"kbs"/a.kb;gt=json.loads((d/"ground_truth.json").read_text());predfile=a.predictions or d/"predictions_abox.json";res=evaluate(json.loads(predfile.read_text()),gt,a.target);out=a.output or d/f"evaluation_{a.target}.json";out.write_text(json.dumps(res,indent=2)+"\n");md=_md_table({a.kb:res["metrics_by_type"]},["jaccard","precision","recall","f1","tbox_recall"]);out.with_suffix(".md").write_text("# Retrieval evaluation\n\n"+md+"\n");print(md)
    if not a.report and not a.kb:p.error("provide --kb or --report")
if __name__=="__main__":main()
