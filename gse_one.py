"""
gse_one.py

Scrape course code, title, and description from:
https://catalog.upenn.edu/courses/educ/

LOCAL VERSION
- No SLURM
- Uses Selenium (headless Firefox)
- Scrapes descriptions by parsing newline-separated text from #textcontainer
- Writes output to current directory
- Prints a visible preview
"""

import re
import json
import time
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ----------------------------
# Patterns
# ----------------------------
# Matches header lines like "EDUC 1000 Foundations of Tutoring Training"
COURSE_HEADER_RE = re.compile(
    r"^(?P<subj>[A-Z]{2,6})\s+(?P<num>\d{4}[A-Z]?)\s+(?P<title>.+)$"
)

# Lines that indicate metadata (not description)
STOP_LINE_RE = re.compile(
    r"^(Fall|Spring|Summer Term|Winter|"
    r"Fall and Spring|Fall or Spring|"
    r"Two Term Class.*|"
    r"\d+\s*-\s*\d+\s*Course Unit(s)?|"
    r"\d+\s*Course Unit(s)?|"
    r"Also Offered As:|Crosslisted As:|"
    r"Prerequisite(s)?:|Permission Required|"
    r"Not Offered Every Year|Formerly|"
    r"View Course Outcomes)$",
    re.IGNORECASE
)

# Sometimes there are headings/boilerplate we don't want to treat as descriptions
SKIP_LINE_RE = re.compile(
    r"^(Education\s+\(EDUC\)|202\d-\d+\s+Catalog|Print Options)$",
    re.IGNORECASE
)


# ----------------------------
# Text normalization
# ----------------------------
def norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")        # nbsp -> space
    s = s.replace("**", "")            # just in case copy/paste includes markdown
    s = re.sub(r"\s+", " ", s).strip() # normalize whitespace
    return s


def extract_courses_from_listing_html(html: str):
    soup = BeautifulSoup(html, "html.parser")

    # UPenn catalog pages: main content is usually here
    container = soup.select_one("#textcontainer") or soup.select_one("main") or soup.body
    if not container:
        return []

    # IMPORTANT: preserve line breaks
    raw_text = container.get_text("\n", strip=True)
    lines = [norm(x) for x in raw_text.splitlines()]
    lines = [x for x in lines if x and not SKIP_LINE_RE.match(x)]

    results = []
    i = 0
    while i < len(lines):
        m = COURSE_HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue

        course_code = f"{m.group('subj')} {m.group('num')}"
        course_title = norm(m.group("title"))

        # Collect description lines until stop markers / next header
        desc_parts = []
        j = i + 1
        while j < len(lines):
            line = lines[j]

            # Stop if next course header
            if COURSE_HEADER_RE.match(line):
                break

            # Stop if metadata line (Fall, 1 Course Unit, etc.)
            if STOP_LINE_RE.match(line):
                break

            desc_parts.append(line)
            j += 1

        description = " ".join(desc_parts).strip()

        results.append(
            {
                "course_code": course_code,
                "course_title": course_title,
                "description": description,
            }
        )

        i = j  # continue from where we stopped

    # Deduplicate (safe)
    dedup = {}
    for r in results:
        dedup[(r["course_code"], r["course_title"])] = r

    return list(dedup.values())


def main():
    url = "https://catalog.upenn.edu/courses/educ/"

    output_json = Path("penn_educ_courses.json")
    output_txt = Path("penn_educ_courses_preview.txt")
    debug_html = Path("penn_educ_page_debug.html")
    debug_lines = Path("penn_educ_lines_debug.txt")

    # Selenium setup (local)
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)

    print(f"\nLoading page: {url}")
    driver.get(url)

    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    time.sleep(1)

    html = driver.page_source
    driver.quit()

    print(f"HTML length: {len(html)}")
    debug_html.write_text(html, encoding="utf-8")
    print(f"Saved HTML debug → {debug_html.resolve()}")

    # Extract courses
    courses = extract_courses_from_listing_html(html)

    # Extra debug: save the exact lines we parsed (super helpful)
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("#textcontainer") or soup.select_one("main") or soup.body
    if container:
        raw_text = container.get_text("\n", strip=True)
        lines = [norm(x) for x in raw_text.splitlines()]
        lines = [x for x in lines if x]
        debug_lines.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved parsed lines debug → {debug_lines.resolve()}")

    # Preview
    print("\n=== SCRAPE PREVIEW ===")
    print(f"Total courses found: {len(courses)}")

    for c in courses[:5]:
        print(f"\n{c['course_code']} — {c['course_title']}")
        if c["description"]:
            print(c["description"][:400] + ("..." if len(c["description"]) > 400 else ""))
        else:
            print("[NO DESCRIPTION FOUND]")

    # Write JSON + TXT
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(courses, f, indent=2, ensure_ascii=False)

    with open(output_txt, "w", encoding="utf-8") as f:
        for c in courses:
            f.write(f"{c['course_code']} — {c['course_title']}\n")
            f.write((c["description"] or "[NO DESCRIPTION FOUND]") + "\n\n")

    print(f"\nSaved JSON → {output_json.resolve()}")
    print(f"Saved TXT preview → {output_txt.resolve()}")
    print("Done ✔")


if __name__ == "__main__":
    main()


