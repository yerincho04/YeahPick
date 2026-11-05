"""One table of every probe result: best-layer accuracy (linear, MLP) and the control ceiling.

Reads outputs/ab_probe/*_probe.json, *_image_probe.json, *_positions.json and writes
outputs/ab_probe/probe_summary.csv and probe_summary.xlsx.
Run from the repo root:  python diagnostics/report/probe_summary_table.py
"""
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "outputs" / "ab_probe"
MODELS = ["llama2-7b", "prismatic-7b", "openvla-base", "openvla-sft", "llama3-8b-instruct", "magma"]
LABELS = {"llama2-7b": "Llama-2-7B", "prismatic-7b": "Prismatic-7B", "openvla-base": "OpenVLA", "openvla-sft": "OpenVLA SFT",
          "llama3-8b-instruct": "Llama-3-8B-Instruct", "magma": "Magma"}
CONDS = {"knowledge": "Knowledge question", "explicit_object": "Named object (control)", "explicit_spatial": "Left/right (control)"}


def row(test, cond, model, r, position="last token"):
    ctrl_keys = [k for k in ("no_question", "shuffled_max", "no_question_mlp", "shuffled_mlp_max") if k in r]
    best = max(range(len(r["question"])), key=lambda l: r["question"][l])
    return {"test": test, "instruction": CONDS[cond], "position": position, "model": LABELS[model],
            "best_layer": best, "linear_%": round(100 * r["question"][best], 1),
            "mlp_%": round(100 * max(r["question_mlp"]), 1) if "question_mlp" in r else "",
            "control_ceiling_%": round(100 * max(max(r[k]) for k in ctrl_keys), 1),
            "above_control": "yes" if r["question"][best] > max(max(r[k]) for k in ctrl_keys) + 0.03 else "no"}


def main():
    rows = []
    for cond in CONDS:
        suffix = "" if cond == "knowledge" else f"-{cond}"
        for test, tag, kind in (("Text only", "", "probe"), ("Image + text (camera frame)", "", "image_probe"),
                                ("Image + text (side by side)", "-sbs", "image_probe")):
            for m in MODELS:
                f = P / f"{m}{suffix}{tag}_{kind}.json"
                if f.exists():
                    rows.append(row(test, cond, m, json.loads(f.read_text())))
    for m in MODELS:
        f = P / f"{m}_positions.json"
        if f.exists():
            r = json.loads(f.read_text())
            for pos, name in (("mean_all", "all tokens (mean)"), ("options", "option-name tokens")):
                rows.append(row("Text only", "knowledge", m, r[pos], name))
    with open(P / "probe_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "probe_summary"
    ws.append(list(rows[0]))
    for r in rows:
        ws.append(list(r.values()))
    for c in ws[1]:
        c.font = Font(bold=True)
    yes, no = PatternFill("solid", start_color="D9EAD3"), PatternFill("solid", start_color="F4CCCC")
    for r in ws.iter_rows(min_row=2):
        r[-1].fill = yes if r[-1].value == "yes" else no
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col, wdt in zip("ABCDEFGHI", (28, 24, 20, 20, 11, 10, 8, 17, 14)):
        ws.column_dimensions[col].width = wdt
    wb.save(P / "probe_summary.xlsx")
    for r in rows:
        print(" | ".join(str(v) for v in r.values()))


if __name__ == "__main__":
    main()
