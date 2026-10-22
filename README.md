# SALT — Side-stem, Attention-gated U-Net, LoRA Tuned

SALT adapts a **frozen** isotropic ViT backbone (DINOv3 ViT-L/16 by default) into a
strong segmentation model for **camouflaged**, **marine-animal**, and **salient**
object detection. It trains only ~5.6M parameters — under 2% of the model (low-rank
adapters plus a light decoder) — and is **RGB-only**: no depth, no auxiliary inputs.

A single RGB pass on a strong isotropic backbone reaches the highest $S$-measure on all
eleven camouflaged, marine, and salient benchmarks, and sets a new state of the art on
both marine-animal benchmarks (**MAS3K mIoU 0.878**). Ten of the eleven are claimed as
wins, nine of them tested per image against released maps; PASCAL-S is an untested tie,
and DUT-OMRON is claimed over SAM2-UNet and BiRefNet but ties SAM3-UNet at matched
training data. The per-image tests and the maps behind them ship below.

> **Why isotropic?** Whether a backbone applies global attention over a
> high-resolution token set predicts whether a low-rank adapter turns resolution into
> accuracy. Isotropic ViTs attend globally over the full grid and keep improving with
> resolution; hierarchical backbones window early attention and pool the grid before
> their global stages, so they plateau earlier. SALT exploits the isotropic regime.

## Architecture

```
input RGB ─┬─> frozen DINOv3 ViT-L  ──► simple feature pyramid ─┐
           │   (+ rank-8 LoRA on attn)   (ViTDet-style 3 levels) │
           │                                                      ▼
           │                                      attention-gated U-Net decoder
           └─> full-res CNN side-stem ───────────► (fused at 3 scales) ──► mask
```

Trainable: LoRA factors on the attention projections, the pyramid neck, the
attention-gated decoder, and the side-stem. Everything in the backbone is frozen.

## Install

```bash
pip install -r requirements.txt
```

The backbone weights download automatically from the Hugging Face Hub on first run
(`camenduru/dinov3-vitl16-pretrain-lvd1689m`; `facebook/dinov2-large` for `--backbone dinov2`).

Training and evaluation require a CUDA GPU. If you hit out-of-memory, lower `--bs`;
for `--bs` below 8 the trainer accumulates gradients to keep the effective batch at 16.

## Data layout

Every train/test directory holds two subfolders with matching, naturally-sortable
filenames:

```
<dataset>/
  RGB/    image001.jpg, image002.jpg, ...
  GT/     image001.png, image002.png, ...   # binary masks
```

This matches the standard COD/SOD/MAS benchmark layout (CAMO, COD10K, CHAMELEON,
NC4K, MAS3K, RMAS, DUTS, ECSSD, ...).

## Train

```bash
python train.py \
  --train_root /data/MAS3K_RMAS_COD \
  --size 336 --bs 32 --epochs 20 \
  --save runs/salt_mas336
```

Useful flags: `--backbone {dinov3,dinov2}`, `--lora_r 8 --lora_alpha 16`,
`--size {336,504,672}`, `--no_sidestem`, `--seed`. Higher `--size` trades compute for
accuracy (see below). The checkpoint (`salt-best.pth`) stores only the trainable
parameters.

## Evaluate

```bash
python eval.py \
  --ckpt runs/salt_mas336/salt-best.pth --size 336 \
  --test_root /data/MAS3K/Test
```

Pass `--test_root` repeatedly to score several sets and print the mean IoU. Reports
S-measure, max/weighted F-measure, mean E-measure, MAE, and mean IoU.

## Use the model directly

```python
import torch
from salt import SALT

model = SALT(img_size=336, lora_r=8, lora_alpha=16, backbone="dinov3").cuda().eval()
model.load_state_dict(torch.load("salt-best.pth"), strict=False)  # frozen backbone keys load from HF

x = torch.randn(1, 3, 336, 336).cuda()      # RGB normalized to mean=std=0.5
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    _, logits = model(x)                      # use the second output as the mask
mask = logits.sigmoid()
```

## Resolution scalability

The backbone runs at `--size`; because the token grid is preserved through every
block, accuracy rises with resolution instead of saturating. On MAS3K:

| size | 336 | 504 | 672 |
|------|-----|-----|-----|
| mIoU | 0.845 | 0.867 | 0.878 |

On MAS3K, resolution is a **training-time** choice: a checkpoint evaluated below its training
resolution scores ~0.01 mIoU under one trained at that size, and gains nothing above it. This
does not transfer to COD, where in a single-seed check the 672-trained model evaluated at 336
beat the 336-trained model by 0.022 mIoU, so check on your own data before reusing a checkpoint
at another size.

## Evaluation protocol, per-image scores, and audits

[`evaluation/`](evaluation/) holds the measurement side of the paper, and runs offline on CPU:

- **`evaluation/map_gate.py`** reproduces the map-validation gate. A per-image comparison is
  only valid if a baseline's released maps back the numbers it is cited for, so every
  comparison is gated on reproducing the published S-measure within 0.005. Of fifteen released
  map sets, nine pass and six fail, one by 0.060.
- **`evaluation/per_image/`** ships the per-image scores behind every comparison the paper
  makes, so the paired bootstrap and Wilcoxon tests rerun with no datasets and no checkpoints:
  `python stat_tests.py compare --a oursdu336_HKU-IS --b sam2unet_HKU-IS`.
- **[`maps/`](maps/)** points at the archived prediction maps
  ([DOI 10.5281/zenodo.23554491](https://doi.org/10.5281/zenodo.23554491)): every map behind the three
  benchmark tables, all fifteen printed rows, ~128,000 PNGs over eleven benchmarks. A paper
  that gates other people's released maps has to release its own to the same standard, so
  each archive records both the printed number and the number recomputed from the shipped
  PNGs. S-measure agrees to four decimals; mIoU and max F-measure agree to within 0.0002.
- **`evaluation/seed_spread.json`** lists all 57 MAS3K cells the paper prints with their
  matched-seed runs. There is no global noise band: each row is read against its own range.
- **`evaluation/dataset_strata.py`** measures what each benchmark rewards, from ground truth
  alone.
- **[`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md)** is the full search, including the five
  additions that did not clear their configuration's seed range.

## Notes

- Default loss is the structure loss (weighted BCE + weighted IoU). `salt/loss.py`
  also provides dice / SSIM / boundary terms, enabled through `CODLoss` weights.
- Inputs of any size are square-resized to `--size` for the backbone; the stretch
  acts as implicit augmentation and outperformed aspect-preserving padding.
- Prediction maps are 8-bit soft and never thresholded, at native ground-truth
  resolution; `eval.py --save_maps` regenerates them.
- This is the clean **method** release. The mechanism-study ablations from the paper
  (pooling / windowed-attention edits, sparse high-resolution refinement) are not
  included here; their measured numbers and seed coverage are in
  `evaluation/seed_spread.json`.

## License

Code is released under the MIT License (see `LICENSE`). Pretrained backbone weights
are downloaded at runtime and are governed by their own licenses.
