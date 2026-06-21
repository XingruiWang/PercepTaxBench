#!/usr/bin/env python3
"""
Generate publication-ready plots for the simulation-image benchmark statistics.

This version packages the dataset statistics directly in the script so it can
run offline without accessing generation_metadata.json.
"""

import os
from pathlib import Path
from textwrap import wrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0) Embedded dataset statistics (simulation images)
# ---------------------------------------------------------------------------
EMBEDDED_SIM_METADATA = {
    "generation_info": {"total_questions": 22644, "total_scenes": 4544},
    "question_type_counts": {
        "affordance_architectural_components_and_fixtures": 321,
        "affordance_art_display_(view_appraise)": 188,
        "affordance_build__span__occupy": 72,
        "affordance_cleaning_and_sanitation": 39,
        "affordance_contain__carry__package": 2,
        "affordance_control__express__light": 72,
        "affordance_display__exhibit__signal_value": 467,
        "affordance_enclosures_and_venues_(enter_use)": 17,
        "affordance_furniture": 443,
        "affordance_grip__carry__operate": 208,
        "affordance_grow__plant_(vegetation)": 15,
        "affordance_household__facility_operations": 287,
        "affordance_mechanical_control": 667,
        "affordance_mediated_action_and_meaning": 47,
        "affordance_operate__use_device": 320,
        "affordance_place__support__work_on": 271,
        "affordance_tableware_and_serveware": 108,
        "affordance_wearables_and_apparel": 8,
        "compositional_set_subtraction_container": 953,
        "compositional_set_subtraction_hollow": 473,
        "counterfactual_heat": 1656,
        "counterfactual_water": 1034,
        "description_matching": 236,
        "function_knowledge": 79,
        "functional_seating": 1488,
        "latent_compressible": 726,
        "latent_containment": 37,
        "material_property": 108,
        "material_scratch_resistance": 664,
        "material_sound_absorption": 100,
        "material_thermal_touch": 676,
        "physical_property": 120,
        "repurposing_bookend_concept": 270,
        "repurposing_container_concept": 1729,
        "repurposing_cushion_concept": 475,
        "repurposing_reflector_concept": 1138,
        "repurposing_shield_concept": 571,
        "repurposing_stepstool_concept": 1581,
        "spatial_above_below": 1231,
        "spatial_closer_to_camera": 1263,
        "spatial_front_behind": 1263,
        "spatial_left_right": 1221,
    },
}

TYPE_CATEGORY_MAP = {
    "affordance_architectural_components_and_fixtures": "taxonomy_description",
    "affordance_art_display_(view_appraise)": "taxonomy_description",
    "affordance_build__span__occupy": "taxonomy_description",
    "affordance_cleaning_and_sanitation": "taxonomy_description",
    "affordance_contain__carry__package": "taxonomy_description",
    "affordance_control__express__light": "taxonomy_description",
    "affordance_display__exhibit__signal_value": "taxonomy_description",
    "affordance_enclosures_and_venues_(enter_use)": "taxonomy_description",
    "affordance_furniture": "taxonomy_description",
    "affordance_grip__carry__operate": "taxonomy_description",
    "affordance_grow__plant_(vegetation)": "taxonomy_description",
    "affordance_household__facility_operations": "taxonomy_description",
    "affordance_mechanical_control": "taxonomy_description",
    "affordance_mediated_action_and_meaning": "taxonomy_description",
    "affordance_operate__use_device": "taxonomy_description",
    "affordance_place__support__work_on": "taxonomy_description",
    "affordance_tableware_and_serveware": "taxonomy_description",
    "affordance_wearables_and_apparel": "taxonomy_description",
    "compositional_set_subtraction_container": "taxonomy_reasoning",
    "compositional_set_subtraction_hollow": "taxonomy_description",
    "counterfactual_heat": "taxonomy_reasoning",
    "counterfactual_water": "taxonomy_reasoning",
    "description_matching": "taxonomy_description",
    "function_knowledge": "taxonomy_description",
    "functional_seating": "taxonomy_reasoning",
    "latent_compressible": "taxonomy_reasoning",
    "latent_containment": "taxonomy_reasoning",
    "material_property": "taxonomy_description",
    "material_scratch_resistance": "taxonomy_reasoning",
    "material_sound_absorption": "taxonomy_reasoning",
    "material_thermal_touch": "taxonomy_reasoning",
    "physical_property": "taxonomy_description",
    "repurposing_bookend_concept": "taxonomy_reasoning",
    "repurposing_container_concept": "taxonomy_reasoning",
    "repurposing_cushion_concept": "taxonomy_reasoning",
    "repurposing_reflector_concept": "taxonomy_reasoning",
    "repurposing_shield_concept": "taxonomy_reasoning",
    "repurposing_stepstool_concept": "taxonomy_reasoning",
    "spatial_above_below": "spatial_relation",
    "spatial_closer_to_camera": "spatial_relation",
    "spatial_front_behind": "spatial_relation",
    "spatial_left_right": "spatial_relation",
}

generation_info = EMBEDDED_SIM_METADATA["generation_info"]
question_type_counts = EMBEDDED_SIM_METADATA["question_type_counts"]

# ---------------------------------------------------------------------------
# 1) Setup & helpers
# ---------------------------------------------------------------------------
OUTDIR = Path(__file__).parent / "plots_sim"
OUTDIR.mkdir(parents=True, exist_ok=True)

CHROME_PATH = Path.home() / ".local/share/plotly/chrome-linux64/chrome"
if CHROME_PATH.exists():
    os.environ.setdefault("KaleidoScope.chromium.binary", str(CHROME_PATH))
    os.environ.setdefault("BROWSER_PATH", str(CHROME_PATH))


def savefig(fname: str) -> None:
    """Save the current Matplotlib figure to PNG and SVG."""
    path_png = OUTDIR / f"{fname}.png"
    path_svg = OUTDIR / f"{fname}.svg"
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.savefig(path_svg, bbox_inches="tight")
    print(f"Saved: {path_png} and {path_svg}")


df_types = (
    pd.DataFrame(
        [
            {
                "type": k,
                "count": v,
                "display_type": k.replace("manual_", "human_level_"),
            }
            for k, v in question_type_counts.items()
        ]
    )
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)

df_cats = (
    pd.DataFrame(
        [{"category": k, "count": v} for k, v in question_type_counts.items()]
    )
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)

CATEGORY_BASE_COLORS = {
    "taxonomy_reasoning": "#F08AA5",
    "attribute_matching": "#FFC857",
    "spatial_relation": "#8FB6FF",
    "description_matching": "#6D597A",
}

DEFAULT_CATEGORY_COLOR = "#888888"
DEFAULT_CATEGORY_ORDER = [
    "taxonomy_reasoning",
    "attribute_matching",
    "spatial_relation",
    "description_matching",
]


def heuristic_category(qtype: str) -> str:
    lowered = qtype.lower()
    if lowered.startswith("spatial_") or lowered.startswith("human_level_spatial"):
        return "spatial_relation"
    if lowered == "description_matching":
        return "description_matching"
    if lowered.startswith("description_") or "taxonomy_description" in lowered:
        return "attribute_matching"
    return "taxonomy_reasoning"


def lighten(color: str, factor: float) -> str:
    rgb = np.array(mpl.colors.to_rgb(color))
    factor = np.clip(factor, 0.0, 1.0)
    return mpl.colors.to_hex(rgb + (1 - rgb) * factor)


df_types["category"] = df_types["type"].map(TYPE_CATEGORY_MAP)

def promote_category(qtype: str, category: str | None) -> str:
    if qtype == "description_matching":
        return "description_matching"
    if category == "taxonomy_description":
        return "attribute_matching"
    if category is None:
        return heuristic_category(qtype)
    return category

df_types["category"] = [
    promote_category(qtype, cat)
    for qtype, cat in zip(df_types["type"], df_types["category"])
]
df_types["category"] = df_types["category"].fillna("taxonomy_reasoning")

df_cats = (
    df_types.groupby("category", as_index=False)["count"]
    .sum()
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)

CATEGORY_ORDER = [
    cat for cat in DEFAULT_CATEGORY_ORDER if cat in df_cats["category"].tolist()
]
for cat in df_cats["category"]:
    if cat not in CATEGORY_BASE_COLORS and cat not in CATEGORY_ORDER:
        CATEGORY_ORDER.append(cat)

CATEGORY_COLORS = {
    category: CATEGORY_BASE_COLORS.get(
        category, mpl.colors.to_hex(mpl.cm.tab20(i % 20))
    )
    for i, category in enumerate(CATEGORY_ORDER)
}

# ---------------------------------------------------------------------------
# 2) Donut Pie (categories)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 7))
cmap = mpl.cm.get_cmap("tab20")
colors = [cmap(i) for i in range(len(df_cats))]

wedges, _ = plt.pie(
    df_cats["count"].values,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.45, edgecolor="white"),
    colors=colors,
)

total = int(df_cats["count"].sum())
for wedge, (category, value) in zip(wedges, zip(df_cats["category"], df_cats["count"])):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x_pos, y_pos = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    label = f"{category}\n{value} ({value / total:.1%})"
    plt.annotate(
        label,
        xy=(x_pos * 0.78, y_pos * 0.78),
        xytext=(x_pos * 1.15, y_pos * 1.15),
        arrowprops=dict(arrowstyle="-", lw=1),
        ha="center",
        va="center",
        fontsize=10,
    )

plt.title("Question Category Distribution (Sim Images)")
plt.tight_layout()
savefig("donut_pie_categories_sim")
plt.close()

# ---------------------------------------------------------------------------
# 3) Top-N question types (horizontal bar)
# ---------------------------------------------------------------------------
TOP_N = 20
top_df = (
    df_types.sort_values("count", ascending=False)
    .head(TOP_N)
    .reset_index(drop=True)
)
top_df["category_rank"] = top_df["category"].map(
    lambda cat: CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
)
top_df = top_df.sort_values(["category_rank", "count"], ascending=[True, False]).reset_index(drop=True)

category_counts = top_df["category"].value_counts()
category_offsets = {cat: 0 for cat in CATEGORY_ORDER}
bar_colors = []
for _, row in top_df.iterrows():
    cat = row["category"]
    idx = category_offsets.get(cat, 0)
    total = category_counts.get(cat, 1)
    base_color = CATEGORY_COLORS.get(cat, DEFAULT_CATEGORY_COLOR)
    shade = lighten(base_color, idx / max(total - 1, 1))
    bar_colors.append(shade)
    category_offsets[cat] = idx + 1
top_df = top_df.drop(columns="category_rank")

plt.figure(figsize=(11, 8))
labels_wrapped = top_df["display_type"].apply(lambda s: "\n".join(wrap(s, 30)))
bars = plt.barh(labels_wrapped, top_df["count"], color=bar_colors)

plt.xlabel("Count")
plt.ylabel("Question Type")
plt.title(f"Top {TOP_N} Question Types (Sim Images)")

for bar, value in zip(bars, top_df["count"].values):
    plt.text(
        bar.get_width() + max(top_df["count"]) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        str(value),
        va="center",
        fontsize=9,
    )

legend_handles = [
    mpl.patches.Patch(
        color=CATEGORY_COLORS.get(cat, DEFAULT_CATEGORY_COLOR),
        label=cat.replace("_", " ").title(),
    )
    for cat in CATEGORY_ORDER
]
plt.legend(handles=legend_handles, loc="lower right", title="Question Category")

plt.tight_layout()
savefig("bar_top_types_sim")
plt.close()

# ---------------------------------------------------------------------------
# 4) Spatial subtypes (bar)
# ---------------------------------------------------------------------------
spatial_keys = ["spatial_above_below", "spatial_front_behind", "spatial_left_right", "spatial_closer_to_camera"]
spatial_vals = [question_type_counts.get(k, 0) for k in spatial_keys]
spatial_labels = ["Above/Below", "Front/Behind", "Left/Right", "Closer/Farther"]

plt.figure(figsize=(8, 6))
x_axis = np.arange(len(spatial_labels))
bars = plt.bar(
    x_axis,
    spatial_vals,
    width=0.6,
    color=[mpl.cm.Set2(i) for i in range(len(spatial_labels))],
)

plt.xticks(x_axis, spatial_labels)
plt.ylabel("Count")
plt.title("Spatial Relation Subtypes (Sim Images)")

for x_coord, y_val in zip(x_axis, spatial_vals):
    plt.text(
        x_coord,
        y_val + max(spatial_vals) * 0.02 if spatial_vals else 0.0,
        str(y_val),
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
savefig("bar_spatial_subtypes_sim")
plt.close()

# ---------------------------------------------------------------------------
# 5) Summary card
# ---------------------------------------------------------------------------
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(8, 4.5))
ax = plt.gca()
ax.axis("off")

box = FancyBboxPatch(
    (0.03, 0.07),
    0.94,
    0.86,
    boxstyle="round,pad=0.02,rounding_size=0.04",
    facecolor=mpl.cm.Pastel1(1),
    edgecolor="none",
)
ax.add_patch(box)

ax.text(
    0.5,
    0.83,
    "Dataset Summary — Sim Images",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
)

total_questions = generation_info.get("total_questions")
total_scenes = generation_info.get("total_scenes")
ax.text(
    0.5,
    0.58,
    f"Total Questions: {total_questions:,}\nTotal Scenes: {total_scenes:,}",
    ha="center",
    va="center",
    fontsize=15,
)

cats_line = ", ".join(df_cats["category"].tolist())
ax.text(
    0.5,
    0.28,
    f"Categories: {cats_line}",
    ha="center",
    va="center",
    fontsize=11,
)

plt.tight_layout()
savefig("summary_card_sim")
plt.close()

print("All sim-image figures generated.")

# ---------------------------------------------------------------------------
# 6) Optional: Plotly Sunburst (Family -> Type)
# ---------------------------------------------------------------------------
try:
    import plotly.express as px  # type: ignore
except ImportError:
    print(
        "Plotly not installed; skipping sunburst. Install with "
        "'pip install plotly kaleido' to enable this figure."
    )
else:
    import re

    def family_of(qtype: str) -> str:
        if qtype.startswith("compositional_"):
            return "compositional"
        if qtype.startswith("manual_taxonomy_"):
            return "human_level_taxonomy"
        if qtype.startswith("material_"):
            return "material"
        if qtype.startswith("functional_"):
            return "functional"
        if qtype.startswith("function_"):
            return "function"
        if qtype.startswith("physical_"):
            return "physical"
        if qtype.startswith("description_"):
            return "description"
        if qtype.startswith("counterfactual_"):
            return "counterfactual"
        if qtype.startswith("repurposing_"):
            return "repurposing"
        if qtype.startswith("latent_"):
            return "latent"
        if qtype.startswith("manual_spatial_"):
            return "human_level_spatial"
        if qtype.startswith("spatial_"):
            return "spatial"
        match = re.match(r"^([A-Za-z0-9]+)_", qtype)
        return match.group(1) if match else "other"

    def pretty_label(value: str) -> str:
        label = value.replace("__", "_")
        label = label.replace("_", " ")
        label = label.replace("  ", " ")
        label = label.replace("compositional set subtraction", "comp. set subtraction")
        label = label.replace("affordance", "afford.")
        label = label.replace("manual taxonomy", "human level tax.")
        label = label.replace("manual spatial", "human level spatial")
        label = label.replace("human_level", "human level")
        return label

    rows = [
        {"family": family_of(k), "type": k.replace("manual_", "human_level_"), "count": v}
        for k, v in question_type_counts.items()
    ]
    sdf = pd.DataFrame(rows)
    sdf_pruned = sdf[sdf["count"] >= 5].copy()
    sdf_pruned["family_label"] = sdf_pruned["family"].map(pretty_label)
    sdf_pruned["type_label"] = sdf_pruned["type"].map(pretty_label)

    fig = px.sunburst(
        sdf_pruned,
        path=["family_label", "type_label"],
        values="count",
        color="family_label",
    )
    fig.update_layout(
        title="Question Type Sunburst (Family → Type) — Sim Images",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    html_path = OUTDIR / "sunburst_question_types_sim.html"
    png_path = OUTDIR / "sunburst_question_types_sim.png"
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), scale=2, width=1000, height=1000)
        print(f"Saved: {html_path} and {png_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Saved interactive sunburst but skipped PNG export due to: {exc}")


