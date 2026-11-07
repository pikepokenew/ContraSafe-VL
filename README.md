# ContraSafe-VL

**A Contrastive Safety Preference Dataset for Vision-Language Models**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2610.23908-b31b1b)](https://arxiv.org/abs/2610.23908)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/DeeWoo/ContraSafe-VL)

> **Paper:** [https://arxiv.org/abs/2610.23908](https://arxiv.org/abs/2610.23908)
> **Dataset:** [https://huggingface.co/datasets/DeeWoo/ContraSafe-VL](https://huggingface.co/datasets/DeeWoo/ContraSafe-VL)

---

## Overview

ContraSafe-VL is a contrastive safety preference dataset for vision-language models (VLMs).
For a **fixed text query**, it provides paired visual contexts whose safety verdicts *flip*:
the same question is unsafe with one image and safe with another. This gives direct
supervision on *how the visual context should change a safety judgment*, going beyond
response-level preference data.

## Dataset statistics

| Item | Value |
|---|---|
| Multimodal queries `(question, image)` | **53,101** |
| Unsafe / Safe | **28,676 / 24,425** |
| Contrastive pairs (unsafe × safe, same text) | **10,115** across 10,022 queries |
| Queries participating in contrastive pairs | 20,230 |
| Additional (non-paired) queries | 32,871 — 18,561 unsafe + 14,310 safe |
| Unique images | 17,862 |
| Response candidates | 9 VLM backbones, 460,529 responses |
| Reward scores | `helpful_score` / `harmless_score` per response |

## Quick start

```python
from datasets import load_dataset, Image

ds = load_dataset("DeeWoo/ContraSafe-VL", split="train")   # 53,101 rows
ds = ds.cast_column("images", [Image()])                    # decode embedded images

row = ds[0]
print(row["question"], row["has_safety_risk"])
row["images"][0].show()

# contrastive pairs (no extra file needed)
pairs = [r for r in ds if r["pair_data_ids"]]
n_pairs = len({tuple(sorted((r["id"], p))) for r in pairs for p in r["pair_data_ids"]})  # 10,115
```

## Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | stable hash `csvl-` + sha1(question + image name)[:12] |
| `question` | string | textual query |
| `images` | list[struct] | `{bytes, path}` per image (HF `Image` compatible) |
| `image_meta` | list[struct] | aligned with `images`: `is_edited`, `source_dataset` |
| `has_safety_risk` | string | `Yes` = this `(question, image)` pair is unsafe, `No` = safe |
| `flip_images` | list[str] | paths of all opposite-label partners |
| `pair_data_id` | string \| null | partner id, set only when `pair_count == 1` |
| `pair_data_ids` | list[str] | ids of all opposite-label partners with the same question |
| `pair_count` | int | number of opposite-label partners |
| `responses` | list[struct] | `model_name`, `response`, `helpful_score`, `harmless_score` |

`image_meta.source_dataset` records the upstream dataset (`spa_vl`, `safe_rlhf_v`,
`datacomp`, `llava`) when applicable.

## Repository structure

```
ContraSafe-VL/
├── README.md            # this file
├── data/                # (optional) processing / construction scripts
└── ...
```

The dataset itself is hosted on Hugging Face (see the badge above); this repository holds
the data construction and analysis code.

## Construction (summary)

1. **Image collection** — images from SPA-VL, SafeRLHF-V, DataComp and LLaVA, plus
   editing-based generation.
2. **Contrastive query construction** — the same text query is paired with visual
   contexts that flip the safety verdict.
3. **Additional query generation** — extra unsafe and safe queries to broaden risk
   categories, visual scenes and query types.
4. **Response preference collection** — nine VLMs generate candidate responses, scored by
   two reward models (`harmless_score`, `helpful_score`).

## Intended use and ethics

This dataset **contains harmful and sensitive images** (violence, illegal acts,
regulated substances, hate, self-harm, etc.) collected for safety research only. It is
released for research on vision-language model safety, red-teaming and alignment.
Do not use it to build or train models for harmful purposes. Access is gated on Hugging
Face; please agree to the terms before use.

## Citation

```bibtex
@article{wu2026contrasafe,
  title   = {ContraSafe-VL: A Contrastive Safety Preference Dataset for Vision-Language Models},
  author  = {Wu, Di and Zhao, Yanyan and Lu, Xin and Yang, Hao and Zhang, Taotao and Zhao, Yunzhen},
  journal = {arXiv preprint arXiv:2610.23908},
  year    = {2026}
}
```

## License

The **code** in this repository is released under the [MIT License](LICENSE).
The **dataset** is released for research use only; please see the
[Hugging Face dataset page](https://huggingface.co/datasets/DeeWoo/ContraSafe-VL)
for its access terms.

## Contact

For questions about the dataset or the paper, please open an issue in this repository.
