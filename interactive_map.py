import json
from pathlib import Path
import os


import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output


# ----------------------------
# Files (local)
# ----------------------------
COORDS_FILE = Path("penn_educ_tsne_coords.json")
COURSES_FILE = Path("penn_educ_courses_with_embeddings.json")

if not COORDS_FILE.exists():
    raise FileNotFoundError(f"Missing {COORDS_FILE.resolve()} (run your t-SNE script first)")
if not COURSES_FILE.exists():
    raise FileNotFoundError(f"Missing {COURSES_FILE.resolve()} (run gse_two.py first)")


# ----------------------------
# Load data
# ----------------------------
coords = json.loads(COORDS_FILE.read_text(encoding="utf-8"))
courses = json.loads(COURSES_FILE.read_text(encoding="utf-8"))

# Lookup by course_code
course_lookup = {}
for c in courses:
    code = (c.get("course_code") or "").strip()
    if not code:
        continue
    course_lookup[code] = {
        "course_code": code,
        "course_title": (c.get("course_title") or "").strip(),
        "description": (c.get("description") or "").strip(),
    }

# Merge coords + metadata
rows = []
for p in coords:
    code = (p.get("code") or "").strip()
    meta = course_lookup.get(code, {})

    title = meta.get("course_title") or (p.get("title") or "").strip()
    desc = meta.get("description") or ""

    rows.append(
        {
            "x": float(p["x"]),
            "y": float(p["y"]),
            "course_code": code,
            "course_title": title,
            "description": desc,
            "label": (p.get("label") or "").strip() or f"{code} — {title}".strip(" —"),
        }
    )

df = pd.DataFrame(rows)

# ----------------------------
# Plotly figure
# ----------------------------
fig = px.scatter(
    df,
    x="x",
    y="y",
    hover_name="course_code",
    hover_data={
        "course_title": True,
        # keep description out of hover so it stays readable; show in panel on click
        "description": False,
        "x": False,
        "y": False,
    },
)

# Make nodes easier to click; store code + title in customdata for callback
fig.update_traces(
    marker=dict(size=10),
    customdata=df[["course_code", "course_title"]].values,
)

fig.update_layout(
    title="UPenn EDUC Courses — Interactive t-SNE Map",
    clickmode="event+select",
    margin=dict(l=20, r=20, t=50, b=20),
)


# ----------------------------
# Dash app
# ----------------------------
app = Dash(__name__)

app.layout = html.Div(
    style={
        "display": "grid",
        "gridTemplateColumns": "2fr 1fr",
        "gap": "16px",
        "height": "95vh",
        "padding": "10px",
        "boxSizing": "border-box",
        "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
    },
    children=[
        html.Div(
            children=[
                dcc.Graph(id="tsne-graph", figure=fig, style={"height": "92vh"})
            ]
        ),
        html.Div(
            id="detail-panel",
            style={
                "border": "1px solid #ddd",
                "borderRadius": "12px",
                "padding": "14px",
                "height": "92vh",
                "overflowY": "auto",
                "background": "white",
            },
            children=[
                html.H3("Click a course", style={"marginTop": 0}),
                html.P(
                    "Click any node to see the course code, title, and description here.",
                    style={"lineHeight": "1.4"},
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("detail-panel", "children"),
    Input("tsne-graph", "clickData"),
)
def show_details(clickData):
    if not clickData:
        return [
            html.H3("Click a course", style={"marginTop": 0}),
            html.P("Click any node to see the course code, title, and description here."),
        ]

    point = clickData["points"][0]
    code, title = point.get("customdata", ["", ""])
    meta = course_lookup.get(code, {})

    # Prefer lookup fields (more reliable), fall back to what was in customdata
    title = meta.get("course_title") or title or "(No title found)"
    desc = meta.get("description") or "(No description found)"

    return [
        html.H3(code or "Unknown course", style={"marginTop": 0}),
        html.H4(title, style={"marginTop": "8px"}),
        html.Hr(),
        html.Div(
            desc,
            style={
                "whiteSpace": "pre-wrap",
                "lineHeight": "1.5",
                "fontSize": "14px",
            },
        ),
    ]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port, debug=False)


