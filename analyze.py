"""Summarize results/raw.jsonl into results/RESULTS.md and results/summary.csv."""
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
CONDITIONS = ["neutral", "pressure", "override"]


def pct(v, n):
    return f"{100 * v / n:.0f}% ({v}/{n})" if n else "n/a"


def main():
    rows = [json.loads(l) for l in (RESULTS / "raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    models = list(dict.fromkeys(r["model"] for r in rows))
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))

    by_mc = defaultdict(lambda: [0, 0])          # (model, condition) -> [violations, valid]
    by_sc = defaultdict(lambda: [0, 0])          # (scenario, condition)
    invalid = defaultdict(int)
    for r in rows:
        if not r["valid"]:
            invalid[r["model"]] += 1
            continue
        for key, table in (((r["model"], r["condition"]), by_mc), ((r["scenario"], r["condition"]), by_sc)):
            table[key][0] += int(r["violation"])
            table[key][1] += 1

    lines = ["# Results", "",
             f"{len(rows)} responses. Violation rate = share of valid answers that broke a safety rule.", "",
             "## Violation rate by model and condition", "",
             "| Model | " + " | ".join(CONDITIONS) + " | Invalid answers |",
             "|---|" + "---|" * (len(CONDITIONS) + 1)]
    for m in models:
        lines.append(f"| {m} | " + " | ".join(pct(*by_mc[(m, c)]) for c in CONDITIONS) + f" | {invalid[m]} |")

    lines += ["", "## Violation rate by scenario (all models)", "",
              "| Scenario | " + " | ".join(CONDITIONS) + " |", "|---|" + "---|" * len(CONDITIONS)]
    for s in scenarios:
        lines.append(f"| {s} | " + " | ".join(pct(*by_sc[(s, c)]) for c in CONDITIONS) + " |")

    lines += ["", "## Example violations", "",
              "Up to 10, with the model's own stated reason. Full text is in raw.jsonl.", ""]
    for r in [r for r in rows if r["violation"]][:10]:
        lines.append(f"- **{r['model']} / {r['scenario']} / {r['condition']}** chose `{r['option']}`: {r['reason']}")

    (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "violations", "valid_answers", "violation_rate"])
        for m in models:
            for c in CONDITIONS:
                v, n = by_mc[(m, c)]
                w.writerow([m, c, v, n, round(v / n, 3) if n else ""])
    print("\n".join(lines[:8 + len(models)]))
    print(f"\nFull summary: {RESULTS / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
