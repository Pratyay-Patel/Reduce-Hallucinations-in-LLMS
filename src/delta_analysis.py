import csv
from collections import defaultdict

RESULTS_PATH = "results/experiment_results.csv"

sample_groups = defaultdict(list)

with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row["nli_support"] = float(row["nli_support"])
        row["hallucination"] = float(row["hallucination"])
        row["compressed"] = int(row["compressed"])
        sample_groups[row["id"]].append(row)

delta_rows = []

for sid, rows in sample_groups.items():
    if len(rows) != 2:
        continue

    uncompressed = next(r for r in rows if r["compressed"] == 0)
    compressed = next(r for r in rows if r["compressed"] == 1)

    delta_nli = compressed["nli_support"] - uncompressed["nli_support"]
    delta_hall = compressed["hallucination"] - uncompressed["hallucination"]

    delta_rows.append((delta_nli, delta_hall))

total = len(delta_rows)

improved = sum(1 for d in delta_rows if d[0] > 0)
worsened = sum(1 for d in delta_rows if d[0] < 0)
unchanged = total - improved - worsened

avg_delta_nli = sum(d[0] for d in delta_rows) / total if total else 0.0

print("\n===== Delta Analysis (Post-Processing) =====")
print(f"Total samples: {total}")
print(f"Improved (NLI ↑): {improved} ({improved/total:.2%})")
print(f"Worsened (NLI ↓): {worsened} ({worsened/total:.2%})")
print(f"Unchanged: {unchanged}")
print(f"Average ΔNLI: {avg_delta_nli:.4f}")