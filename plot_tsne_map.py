import json
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# CONFIG
# =========================
INPUT_COORDS = Path("penn_educ_tsne_coords.json")
OUTPUT_PNG = Path("penn_educ_tsne_map.png")

FIGSIZE = (12, 9)
POINT_SIZE = 40
LABEL_EVERY_N = 5   # label every Nth point to avoid clutter (set to 1 to label all)
ALPHA = 0.75


# =========================
# LOAD COORDS
# =========================
if not INPUT_COORDS.exists():
    raise FileNotFoundError(f"Could not find {INPUT_COORDS.resolve()}")

with open(INPUT_COORDS, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list) or len(data) == 0:
    raise ValueError("Coordinate file is empty or malformed")

xs = [d["x"] for d in data]
ys = [d["y"] for d in data]

labels = [
    d.get("label")
    or d.get("title")
    or d.get("code", "")
    for d in data
]

print(f"Loaded {len(xs)} points")


# =========================
# PLOT
# =========================
plt.figure(figsize=FIGSIZE)

plt.scatter(xs, ys, s=POINT_SIZE, alpha=ALPHA)

# Label a subset of points for readability
for i, label in enumerate(labels):
    if LABEL_EVERY_N > 0 and i % LABEL_EVERY_N != 0:
        continue
    if not label:
        continue
    plt.text(
        xs[i],
        ys[i],
        label,
        fontsize=8,
        alpha=0.8
    )

plt.title("UPenn EDUC Courses — t-SNE Map (SBERT embeddings)")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200)
plt.show()

print(f"Saved plot → {OUTPUT_PNG.resolve()}")
