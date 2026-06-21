#!/usr/bin/env python3
"""
Generate publication-ready plots for the real-image benchmark statistics.

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
# 0) Embedded dataset statistics (real images)
# ---------------------------------------------------------------------------
EMBEDDED_REAL_METADATA = {
    "generation_info": {"total_questions": 5439, "total_scenes": 1258},
    "question_type_counts": {
        "affordance_architectural_components_and_fixtures": 11,
        "affordance_art_display_(view_appraise)": 9,
        "affordance_build__span__occupy": 94,
        "affordance_cleaning_and_sanitation": 2,
        "affordance_contain__carry__package": 74,
        "affordance_control__express__light": 10,
        "affordance_display__exhibit__signal_value": 46,
        "affordance_enclosures_and_venues_(enter_use)": 53,
        "affordance_food_—_ingredients_and_produce": 21,
        "affordance_food_—_prepared_dishes": 30,
        "affordance_furniture": 123,
        "affordance_grip__carry__operate": 121,
        "affordance_grow__plant_(vegetation)": 60,
        "affordance_household__facility_operations": 10,
        "affordance_interact_with_living_moving_things": 66,
        "affordance_mechanical_control": 288,
        "affordance_mediated_action_and_meaning": 13,
        "affordance_operate__use_device": 94,
        "affordance_place__support__work_on": 179,
        "affordance_sit__ride__attend": 129,
        "affordance_structured_operational_engagement": 9,
        "affordance_tableware_and_serveware": 57,
        "affordance_wearables_and_apparel": 284,
        "compositional_set_subtraction_container": 316,
        "compositional_set_subtraction_hollow": 85,
        "counterfactual_heat": 247,
        "counterfactual_water": 57,
        "description_matching": 255,
        "function_knowledge": 299,
        "functional_foldable": 178,
        "functional_seating": 170,
        "latent_compressible": 293,
        "latent_containment": 37,
        "manual_spatial_relation": 13,
        "manual_taxonomy_description": 21,
        "manual_taxonomy_reasoning": 16,
        "material_property": 232,
        "material_scratch_resistance": 105,
        "material_sound_absorption": 64,
        "material_thermal_touch": 112,
        "physical_property": 19,
        "repurposing_bookend_concept": 8,
        "repurposing_container_concept": 122,
        "repurposing_cushion_concept": 48,
        "repurposing_lever_concept": 11,
        "repurposing_reflector_concept": 319,
        "repurposing_shield_concept": 44,
        "repurposing_stepstool_concept": 118,
        "spatial_above_below": 226,
        "spatial_front_behind": 145,
        "spatial_left_right": 96,
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
    "affordance_food_—_ingredients_and_produce": "taxonomy_description",
    "affordance_food_—_prepared_dishes": "taxonomy_description",
    "affordance_furniture": "taxonomy_description",
    "affordance_grip__carry__operate": "taxonomy_description",
    "affordance_grow__plant_(vegetation)": "taxonomy_description",
    "affordance_household__facility_operations": "taxonomy_description",
    "affordance_interact_with_living_moving_things": "taxonomy_description",
    "affordance_mechanical_control": "taxonomy_description",
    "affordance_mediated_action_and_meaning": "taxonomy_description",
    "affordance_operate__use_device": "taxonomy_description",
    "affordance_place__support__work_on": "taxonomy_description",
    "affordance_sit__ride__attend": "taxonomy_description",
    "affordance_structured_operational_engagement": "taxonomy_description",
    "affordance_tableware_and_serveware": "taxonomy_description",
    "affordance_wearables_and_apparel": "taxonomy_description",
    "compositional_set_subtraction_container": "taxonomy_reasoning",
    "compositional_set_subtraction_hollow": "taxonomy_description",
    "counterfactual_heat": "taxonomy_reasoning",
    "counterfactual_water": "taxonomy_reasoning",
    "description_matching": "taxonomy_description",
    "function_knowledge": "taxonomy_description",
    "functional_foldable": "taxonomy_reasoning",
    "functional_seating": "taxonomy_reasoning",
    "latent_compressible": "taxonomy_reasoning",
    "latent_containment": "taxonomy_reasoning",
    "manual_spatial_relation": "spatial_relation",
    "manual_taxonomy_description": "taxonomy_description",
    "manual_taxonomy_reasoning": "taxonomy_reasoning",
    "material_property": "taxonomy_description",
    "material_scratch_resistance": "taxonomy_reasoning",
    "material_sound_absorption": "taxonomy_reasoning",
    "material_thermal_touch": "taxonomy_reasoning",
    "physical_property": "taxonomy_description",
    "repurposing_bookend_concept": "taxonomy_reasoning",
    "repurposing_container_concept": "taxonomy_reasoning",
    "repurposing_cushion_concept": "taxonomy_reasoning",
    "repurposing_lever_concept": "taxonomy_reasoning",
    "repurposing_reflector_concept": "taxonomy_reasoning",
    "repurposing_shield_concept": "taxonomy_reasoning",
    "repurposing_stepstool_concept": "taxonomy_reasoning",
    "spatial_above_below": "spatial_relation",
    "spatial_front_behind": "spatial_relation",
    "spatial_left_right": "spatial_relation",
}

generation_info = EMBEDDED_REAL_METADATA["generation_info"]
question_type_counts = EMBEDDED_REAL_METADATA["question_type_counts"]

# ---------------------------------------------------------------------------
# 1) Setup & helpers
# ---------------------------------------------------------------------------
OUTDIR = Path(__file__).parent / "plots_real"
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

df_types["category"] = df_types["type"].map(TYPE_CATEGORY_MAP)
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

plt.title("Question Category Distribution (Real Images)")
plt.tight_layout()
savefig("donut_pie_categories_real")
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
top_df = top_df.sort_values(["category_rank", "count"], ascending=[True, False])
top_df = top_df.reset_index(drop=True)

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
plt.title(f"Top {TOP_N} Question Types (Real Images)")

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
savefig("bar_top_types_real")
plt.close()

# ---------------------------------------------------------------------------
# 4) Spatial subtypes (bar)
# ---------------------------------------------------------------------------
spatial_keys = ["spatial_above_below", "spatial_front_behind", "spatial_left_right"]
spatial_vals = [question_type_counts.get(k, 0) for k in spatial_keys]
spatial_labels = ["Above/Below", "Front/Behind", "Left/Right"]

plt.figure(figsize=(7, 6))
x_axis = np.arange(len(spatial_labels))
bars = plt.bar(
    x_axis,
    spatial_vals,
    width=0.6,
    color=[mpl.cm.Set2(i) for i in range(len(spatial_labels))],
)

plt.xticks(x_axis, spatial_labels)
plt.ylabel("Count")
plt.title("Spatial Relation Subtypes (Real Images)")

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
savefig("bar_spatial_subtypes_real")
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
    facecolor=mpl.cm.Pastel1(0),
    edgecolor="none",
)
ax.add_patch(box)

ax.text(
    0.5,
    0.83,
    "Dataset Summary — Real Images",
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

cats_line = ", ".join(category_counts.keys())
ax.text(
    0.5,
    0.28,
    f"Categories: {cats_line}",
    ha="center",
    va="center",
    fontsize=11,
)

plt.tight_layout()
savefig("summary_card_real")
plt.close()

print("All real-image figures generated.")

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
        title="Question Type Sunburst (Family → Type) — Real Images",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    html_path = OUTDIR / "sunburst_question_types_real.html"
    png_path = OUTDIR / "sunburst_question_types_real.png"
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), scale=2, width=1000, height=1000)
        print(f"Saved: {html_path} and {png_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Saved interactive sunburst but skipped PNG export due to: {exc}")


