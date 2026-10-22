"""Evaluate a trained SALT checkpoint with the standard COD/SOD metrics
(S-measure, max F-measure, weighted F-measure, mean E-measure, MAE, mean IoU).

Example:
  python eval.py --ckpt runs/salt_mas336/salt-best.pth --size 336 \
      --test_root /data/MAS3K/Test

Each test dir must contain RGB/ and GT/. Pass --test_root multiple times to
evaluate several sets and report the per-set numbers plus the mean IoU.
Requires `pip install pysodmetrics`.

`--save_maps DIR` writes the prediction maps alongside the metrics, as 8-bit soft
PNGs at ground-truth resolution, one subdirectory per test set. This is the code path
that produced the released map sets (see maps/README.md); the maps are not thresholded,
because a binarized map cannot be rescored for S-measure or max F-measure.
"""
import json
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from salt import SALT, TestDataset


def evaluate(model, root, size, save_maps=None):
    import py_sod_metrics
    from py_sod_metrics import Smeasure, Emeasure, MAE, WeightedFmeasure
    loader = TestDataset(root, size)
    if save_maps:
        os.makedirs(save_maps, exist_ok=True)
    SM, EM, MA, WF = Smeasure(), Emeasure(), MAE(), WeightedFmeasure()
    FM = py_sod_metrics.FmeasureV2(metric_handlers={
        "fm": py_sod_metrics.FmeasureHandler(with_adaptive=True, with_dynamic=True, beta=0.3),
        "iou": py_sod_metrics.IOUHandler(with_adaptive=True, with_dynamic=True)})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(loader.length):
            x, gt, name = loader.load_data(); x = x.cuda()
            _, pred = model(x)
            # sigmoid and the upsample stay under bf16 autocast: this is the path the
            # published numbers came from, and promoting to fp32 here moves mIoU ~0.0015.
            pred = torch.sigmoid(pred)
            pred = F.interpolate(pred, size=gt.shape, mode='bilinear', align_corners=False)
            pred = pred.float().cpu().squeeze().numpy()
            FM.step(pred=pred.astype(np.float64), gt=gt.astype(np.float64))
            p8 = np.clip(pred * 255, 0, 255).astype(np.uint8)
            SM.step(pred=p8, gt=gt); EM.step(pred=p8, gt=gt)
            MA.step(pred=p8, gt=gt); WF.step(pred=p8, gt=gt)
            if save_maps:
                Image.fromarray(p8, mode="L").save(
                    os.path.join(save_maps, os.path.splitext(name)[0] + ".png"), compress_level=6)
    return {"mIoU": float(FM.get_results()["iou"]["dynamic"].mean()),
            "Smeasure": float(SM.get_results()["sm"]),
            "maxFm": float(FM.get_results()["fm"]["dynamic"].max()),
            "w_Fm": float(WF.get_results()["wfm"]),
            "meanEm": float(EM.get_results()["em"]["curve"].mean()),
            "MAE": float(MA.get_results()["mae"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_root", required=True, action="append",
                    help="dir with RGB/ and GT/; repeat for multiple sets")
    ap.add_argument("--backbone", default="dinov3", choices=["dinov3", "dinov2"])
    ap.add_argument("--size", type=int, default=336)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--no_sidestem", action="store_true")
    ap.add_argument("--out", default=None, help="optional path to dump results JSON")
    ap.add_argument("--save_maps", default=None,
                    help="write prediction maps here, one subdir per --test_root")
    args = ap.parse_args()

    model = SALT(img_size=args.size, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                 use_sidestem=not args.no_sidestem, backbone=args.backbone).cuda()
    sd = torch.load(args.ckpt, map_location="cuda")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Only the frozen backbone weights should be "missing" (they load from HF).
    print(f"loaded {args.ckpt}: {len(unexpected)} unexpected keys", flush=True)
    model.eval()

    per_set = {}
    for root in args.test_root:
        sub = (os.path.join(args.save_maps, os.path.basename(root.rstrip("/\\")))
               if args.save_maps else None)
        per_set[root] = evaluate(model, root, args.size, save_maps=sub)
    for name, r in per_set.items():
        print(f"\n[{name}] " + ", ".join(f"{k}:{v:.4f}" for k, v in r.items()), flush=True)
    if len(per_set) > 1:
        mean_miou = float(np.mean([r["mIoU"] for r in per_set.values()]))
        print(f"\nmean mIoU over {len(per_set)} sets = {mean_miou:.4f}", flush=True)
    if args.out:
        json.dump(per_set, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
