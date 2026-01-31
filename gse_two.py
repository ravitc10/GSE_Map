import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ========================================
# LOCAL CONFIG
# ========================================
INPUT_FILE = Path("penn_educ_courses.json")
OUTPUT_FILE = Path("penn_educ_courses_with_embeddings.json")

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64
MAX_LENGTH = 256  # longer descriptions benefit from >128


# ========================================
# Text helper
# ========================================
def build_text(course: dict) -> str:
    """
    Adapted to your scraped output:
      - course_code
      - course_title
      - description
    Creates a single string for embedding.
    """
    code = (course.get("course_code") or "").strip()
    title = (course.get("course_title") or "").strip()
    desc = (course.get("description") or "").strip()

    # If description is missing, still embed code+title (but you can skip if you prefer)
    parts = []
    if code:
        parts.append(code)
    if title:
        parts.append(title)
    header = " ".join(parts).strip()

    if desc and header:
        return f"{header}. {desc}"
    if desc:
        return desc
    return header


# ========================================
# SBERT SETUP (mean pooling)
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(SBERT_MODEL_NAME)
base_model = AutoModel.from_pretrained(SBERT_MODEL_NAME).to(device)
base_model.eval()


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * input_mask_expanded).sum(dim=1)
    counts = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def sbert_generate_embeddings_batch(texts):
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        outputs = base_model(**encoded)
        pooled = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])

        # L2 normalize so cosine similarity is straightforward later
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        all_embeddings.append(pooled.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings.tolist()


# ========================================
# MAIN
# ========================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing {INPUT_FILE.resolve()}\n"
        f"Run gse_one.py first to generate penn_educ_courses.json."
    )

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    courses = json.load(f)

if not isinstance(courses, list):
    raise ValueError("Expected a list of course dicts in penn_educ_courses.json")

texts = []
valid_indices = []

for idx, course in enumerate(courses):
    text = build_text(course).strip()
    # Prefer skipping if truly empty
    if text:
        texts.append(text)
        valid_indices.append(idx)

print(f"Loaded courses: {len(courses)}")
print(f"Embedding texts: {len(texts)}")

if texts:
    embeddings = sbert_generate_embeddings_batch(texts)

    for emb_idx, course_idx in enumerate(tqdm(valid_indices, desc="Attaching embeddings")):
        emb = embeddings[emb_idx]
        courses[course_idx]["embedding"] = [round(x, 8) for x in emb]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(courses, f, indent=2, ensure_ascii=False)

with_embeddings = sum(1 for c in courses if "embedding" in c)
print(f"\nSaved: {OUTPUT_FILE.resolve()}")
print(f"Courses with embeddings: {with_embeddings}/{len(courses)}")

# Preview 1 record
for c in courses:
    if "embedding" in c:
        print("\nSample:")
        print(c.get("course_code"), "-", c.get("course_title"))
        print("Embedding length:", len(c["embedding"]))
        break
