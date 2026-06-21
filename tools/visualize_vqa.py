"""
Visualize VQA samples from taxonomyQABench_realimage.
Each panel shows the image with question, answer, and category overlaid.

Usage:
    python visualize_vqa.py                        # random 12 samples, all categories
    python visualize_vqa.py --n 20                 # 20 random samples
    python visualize_vqa.py --category spatial     # filter by category
    python visualize_vqa.py --indices 0 5 10       # specific question_index values
    python visualize_vqa.py --out my_vis.png       # custom output path
    python visualize_vqa.py --cols 4               # 4 columns per row
"""

import argparse
import json
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

DATA_ROOT = Path("/path/to/Taxonomy/Data/realImage/taxonomyQABench_realimage")
DATA_ROOT = Path("/path/to/Taxonomy/Data/realImage/taxonomyQABench_realimage")
IMAGES_DIR = DATA_ROOT / "images"
JSON_PATH = DATA_ROOT / "all_questions.json"

CATEGORY_COLORS = {
    "spatial":        "#4e79a7",
    "capability":     "#f28e2b",
    "material":       "#e15759",
    "function":       "#76b7b2",
    "compositional":  "#59a14f",
    "counterfactual": "#edc948",
    "repurposing":    "#b07aa1",
    "affordance":     "#ff9da7",
    "description":    "#9c755f",
    "latent":         "#bab0ac",
}


def wrap_text(text: str, width: int = 55) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def load_data():
    with open(JSON_PATH) as f:
        return json.load(f)


def plot_samples(samples: list, cols: int = 4, out_path: str = "visualization.png"):
    n = len(samples)
    rows = (n + cols - 1) // cols

    fig_w = cols * 4
    fig_h = rows * 7
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, sample in zip(axes, samples):
        img_path = IMAGES_DIR / sample["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            ax.text(0.5, 0.5, f"Image not found:\n{img_path.name}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.axis("off")
            continue

        ax.imshow(img)
        ax.axis("off")

        # Category badge (top-left)
        cat = sample.get("question_category", "unknown")
        color = CATEGORY_COLORS.get(cat, "#888888")
        ax.text(0.02, 0.98, cat.upper(),
                transform=ax.transAxes,
                fontsize=7, fontweight="bold", color="white",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85, edgecolor="none"))

        # Question + Answer below image (ax.text with clip_on=False, axis("off") hides xlabel)
        ans = sample.get("answer", "")
        ans_obj = sample.get("answer_object", "")
        display_ans = ans + (f"  [{ans_obj}]" if ans_obj and ans_obj != ans else "")
        q_text = wrap_text("Q: " + sample["question"], width=50)
        a_text = f"A: {display_ans}"
        id_text = f"[{sample.get('question_index', '')}] {sample.get('image_path', '')}"
        ax.text(0.0, -0.02, f"{id_text}\n{q_text}\n{a_text}",
                transform=ax.transAxes,
                fontsize=7.5, color="#222222",
                va="top", ha="left",
                clip_on=False)

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    # Category legend
    legend_handles = [
        mpatches.Patch(color=c, label=k)
        for k, c in CATEGORY_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=8, title="Category",
               bbox_to_anchor=(0.5, -0.01),
               frameon=True, edgecolor="#cccccc")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize VQA samples from taxonomyQABench_realimage")
    parser.add_argument("--n", type=int, default=12, help="Number of random samples (default: 12)")
    parser.add_argument("--category", type=str, default=None,
                        choices=list(CATEGORY_COLORS.keys()),
                        help="Filter by question_category")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Specific question_index values to display")
    parser.add_argument("--cols", type=int, default=4, help="Columns per row (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out", type=str, default="visualization.png",
                        help="Output image path (default: visualization.png)")
    args = parser.parse_args()

    data = load_data()
    print(f"Loaded {len(data)} samples.")

    if args.category:
        data = [d for d in data if d["question_category"] == args.category]
        print(f"After category filter '{args.category}': {len(data)} samples.")

    if args.indices is not None:
        idx_set = set(args.indices)
        samples = [d for d in data if d["question_index"] in idx_set]
        if not samples:
            print("No matching indices found.")
            return
    else:
        random.seed(args.seed)
        samples = random.sample(data, min(args.n, len(data)))

    print(f"Visualizing {len(samples)} samples ({args.cols} columns).")
    plot_samples(samples, cols=args.cols, out_path=args.out)


if __name__ == "__main__":
    main()
