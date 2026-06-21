# TaxonomyVQA — Data Processing & Benchmark Generation

Code for building **TaxonomyVQA**, a visual-question-answering benchmark that probes
vision-language models on physical object properties (material, shape, function,
affordance), spatial relations, and compositional / counterfactual reasoning over
both **real images** (OpenImages) and **simulated scenes**.

This repository contains the **data-processing and benchmark-generation pipeline**.
The released VQA data lives on Hugging Face:

- 📦 **Benchmark (real + sim, open-ended):** `TaxonomyProject/TaxonomyVQA`
- 📦 **Simulation metadata (taxonomy, objects, scene annotations):** [`TaxonomyProject/SimulationMetadata`](https://huggingface.co/datasets/TaxonomyProject/SimulationMetadata)

The 3D-annotation / detection stages build on
[SpatialReasonerDataGen](https://github.com/wufeim/SpatialReasonerDataGen)
(SpatialReasoner, Ma et al. 2025).

---

## Pipeline overview

```
Real images (OpenImages)  ─┐
                           ├─►  3D annotation  ─►  object description  ─►  QA generation  ─►  open-ended benchmark  ─►  TSV / HF dataset
Simulated scenes          ─┘   (pipeline/3d_annotation,   (pipeline/object_description)   (pipeline/qa_gen)        (create_open_answer_*.py)   (evaluation/)
                                pipeline/object_detection)
```

| Stage | Directory | Entry points |
|-------|-----------|--------------|
| 2D detection + segmentation | `pipeline/object_detection/` | `run_object_detection.py`, `hybrid_detector.py` |
| 3D ground-truth annotation | `pipeline/3d_annotation/` | `generate_3d_groundtruth_production.py` |
| Object property descriptions (Gemini) | `pipeline/object_description/` | `scripts/python/generate_image_object_descriptions.py` |
| QA generation (real / sim) | `pipeline/qa_gen/` | `scripts/core/generate_taxonomyqabench_realimage.py`, `…_simimage.py` |
| Open-ended conversion | `pipeline/qa_gen/` | `create_open_answer_benchmark.py`, `create_open_answer_gt_benchmark.py` |
| Aggregate → TSV | `pipeline/qa_gen/` | `scripts/core/aggregate_unified_qa.py` |
| Evaluation (VLMEvalKit) | `evaluation/` | `taxonomy.py`, `utils_taxonomy.py`, `run.py` |
| Image rendering / placement (sim) | `pipeline/image_render/`, `pipeline/object_placement/` | `gen_scene_structure.py`, `filter_placable_object.py` |
| Helpers | `tools/` | `visualize_vqa.py`, human-filtering Gradio apps, HF upload |

See `docs/` for detailed write-ups: `QA_GENERATION_README.md`,
`SPATIALREASONER_DATAGEN_PIPELINE_ARCHITECTURE.md`, `UNIFIED_PIPELINE_README.md`,
and `pipeline/qa_gen/*.md`.

---

## Setup

```bash
git clone https://github.com/XingruiWang/TaxonomyVQA
cd TaxonomyVQA
pip install -r requirements.txt          # core (QA gen, eval, tooling)
# GPU stages (3D annotation / detection) need extra deps — see docs/INSTALL.md
```

### API keys & paths
Several stages call the **Google Gemini** API. Provide a key via environment variable
(scripts read `GEMINI_API_KEY` / `GOOGLE_API_KEY`); never hard-code keys.

```bash
cp .env.example .env        # then edit
export GEMINI_API_KEY=...    # or set in your shell / SLURM script
```

All paths in scripts use `/path/to/...` placeholders — edit them, or set
`TAXONOMY_DATA_ROOT`, to point at your local data.

---

## Evaluation

The `evaluation/` files are a thin integration for
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Drop `taxonomy.py` into
`vlmeval/dataset/` and `utils_taxonomy.py` into `vlmeval/dataset/utils/`, register the
dataset, then run with `run.py`. The benchmark TSVs are produced by
`pipeline/qa_gen/scripts/core/aggregate_unified_qa.py`.

---

## Known issues

A few files inherited from upstream/WIP code currently have pre-existing syntax errors
and are included for reference only:
`pipeline/object_detection/gemini_object_detection.py`,
`pipeline/qa_gen/multiuser_app_vN.py`,
`pipeline/qa_gen/scripts/modules/qa_modules/cot_reasoning_utils.py`.

---

## License

Released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), consistent with
the upstream SpatialReasonerDataGen. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{ma2025spatialreasoner,
  title={SpatialReasoner: Towards Explicit and Generalizable 3D Spatial Reasoning},
  author={Ma, Wufei and Chou, Yu-Cheng and Liu, Qihao and Wang, Xingrui and de Melo, Celso and Xie, Jianwen and Yuille, Alan},
  journal={arXiv preprint arXiv:2504.20024},
  year={2025}
}
```

> _Citation for the TaxonomyVQA paper will be added here upon publication._
