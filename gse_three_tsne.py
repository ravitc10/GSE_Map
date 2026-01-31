import json
import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial import cKDTree
from tqdm import tqdm


# =========================
# LOCAL CONFIG
# =========================
INPUT_PATH = "penn_educ_courses_with_embeddings.json"
OUTPUT_FRONTEND = "penn_educ_tsne_coords.json"
# Optional second output (same content, different indent)
OUTPUT_BACKEND = ""  # e.g., "penn_educ_tsne_coords_backend.json"

# t-SNE + overlap handling params (local defaults)
PERPLEXITY = 30.0
MIN_DIST = 1.0
MAGNIFICATION = 2.0
JITTER_RADIUS = 0.01
RANDOM_SEED = 42


# =========================
# Overlap separation (your function, unchanged)
# =========================
def separate_overlapping_points(coords, similarity_matrix, course_keys,
                                min_dist=1.0, magnification=2.0, jitter_radius=0.01, random_seed=42):
    """
    Fast approach: magnify coordinates and add jitter to exact duplicates.
    Args:
        coords: numpy array of shape (n, 2) containing the coordinates
        similarity_matrix: numpy array of shape (n, n) containing similarity scores
        course_keys: list of tuples corresponding to each point (kept for compatibility)
        min_dist: minimum distance threshold for considering points as duplicates
        magnification: factor to scale up all coordinates (magnifies differences)
        jitter_radius: maximum radius for random jitter applied to exact duplicates
        random_seed: random seed for reproducibility
    """
    n = coords.shape[0]
    print(f"Magnifying coordinates by factor {magnification}...")

    coords_scaled = coords * magnification

    print("Finding exact duplicates...")
    tree = cKDTree(coords_scaled)
    duplicate_threshold = min_dist * magnification * 0.1
    pairs = tree.query_pairs(duplicate_threshold, output_type='set')

    # In the old pipeline you kept together near-identical across semesters.
    # We don't have semesters here, so keep_together stays empty.
    keep_together = set()

    print(f"Applying jitter to {len(pairs) - len(keep_together)} duplicate pairs...")
    np.random.seed(random_seed)
    jittered = np.zeros(n, dtype=bool)

    for i, j in pairs:
        if (i, j) in keep_together:
            continue
        if not jittered[i] and not jittered[j]:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, jitter_radius * magnification)
            jitter = radius * np.array([np.cos(angle), np.sin(angle)])

            coords_scaled[i] += jitter
            coords_scaled[j] -= jitter
            jittered[i] = True
            jittered[j] = True

    print("Handling remaining exact coordinate matches...")
    step = jitter_radius * magnification * 0.1
    if step <= 0:
        step = 1e-6

    rounded_coords = np.round(coords_scaled / step) * step
    unique_coords, inverse, counts = np.unique(rounded_coords, axis=0, return_inverse=True, return_counts=True)

    for idx in np.where(counts > 1)[0]:
        group_indices = np.where(inverse == idx)[0]
        n_in_group = len(group_indices)
        if n_in_group > 1:
            center = coords_scaled[group_indices[0]].copy()
            angles = np.linspace(0, 2 * np.pi, n_in_group, endpoint=False)
            for k, gi in enumerate(group_indices):
                angle = angles[k]
                offset = jitter_radius * magnification * np.array([np.cos(angle), np.sin(angle)])
                coords_scaled[gi] = center + offset

    print(f"Completed coordinate separation. Final coordinate range: "
          f"x=[{coords_scaled[:, 0].min():.4f}, {coords_scaled[:, 0].max():.4f}], "
          f"y=[{coords_scaled[:, 1].min():.4f}, {coords_scaled[:, 1].max():.4f}]")

    return coords_scaled


# =========================
# Load embeddings output
# =========================
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Keep only rows that have embeddings
courses = []
for c in data:
    emb = c.get("embedding")
    if isinstance(emb, list) and len(emb) > 0:
        courses.append(c)

if not courses:
    raise RuntimeError(
        f"No embeddings found in {INPUT_PATH}. "
        f"Make sure gse_two.py ran successfully and added 'embedding' to courses."
    )

labels = [
    f"{c.get('course_code','').strip()} — {c.get('course_title','').strip()}".strip(" —")
    for c in courses
]
codes = [c.get("course_code", "").strip() for c in courses]
titles = [c.get("course_title", "").strip() for c in courses]

X = np.array([c["embedding"] for c in courses], dtype=np.float32)

n, dim = X.shape
print(f"Loaded {n} courses with embeddings (dim={dim}).")

# Safety: ensure embeddings are normalized (gse_two.py normalizes, but we re-normalize defensively)
norms = np.linalg.norm(X, axis=1, keepdims=True)
norms[norms == 0] = 1.0
X = X / norms


# =========================
# Build similarity + distance matrices
# =========================
print("Computing cosine similarity matrix...")
sim_matrix = X @ X.T
sim_matrix = np.clip(sim_matrix, 0.0, 1.0)  # cosine should be [-1,1], but your embeddings are usually >=0-ish; clip for safety
np.fill_diagonal(sim_matrix, 1.0)

print("Converting similarity to distance matrix...")
dist_matrix = 1.0 - sim_matrix
np.fill_diagonal(dist_matrix, 0.0)
dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
assert np.all(dist_matrix >= 0), "Distance matrix has negative values!"


# =========================
# Run t-SNE
# =========================
# t-SNE perplexity must be < n (rough rule of thumb: <= (n-1)/3 is safer)
if n < 3:
    raise RuntimeError("Need at least 3 courses with embeddings to run t-SNE.")

safe_perplexity = min(PERPLEXITY, max(2.0, (n - 1) / 3.0))
if safe_perplexity != PERPLEXITY:
    print(f"Adjusting perplexity from {PERPLEXITY} -> {safe_perplexity:.2f} because n={n}")

print(f"Running t-SNE on {n} courses (perplexity={safe_perplexity:.2f})...")
tsne = TSNE(
    n_components=2,
    metric="precomputed",
    random_state=RANDOM_SEED,
    perplexity=safe_perplexity,
    init="random",
)
coords = tsne.fit_transform(dist_matrix)

print(f"Separating overlaps (min_dist={MIN_DIST}, magnification={MAGNIFICATION}, jitter_radius={JITTER_RADIUS})...")
course_keys = [("unknown", code) for code in codes]  # keep same tuple structure as your original pipeline
coords = separate_overlapping_points(
    coords,
    sim_matrix,
    course_keys,
    min_dist=MIN_DIST,
    magnification=MAGNIFICATION,
    jitter_radius=JITTER_RADIUS,
    random_seed=RANDOM_SEED,
)

# =========================
# Save output (frontend-friendly)
# =========================
output = []
for i in range(n):
    x, y = coords[i]
    output.append({
        "code": codes[i],
        "title": titles[i],
        "label": labels[i],
        "x": float(x),
        "y": float(y),
    })

with open(OUTPUT_FRONTEND, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

if OUTPUT_BACKEND:
    with open(OUTPUT_BACKEND, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

print(f"Saved t-SNE coordinates for {n} courses to {OUTPUT_FRONTEND}"
      + (f" and {OUTPUT_BACKEND}" if OUTPUT_BACKEND else ""))
