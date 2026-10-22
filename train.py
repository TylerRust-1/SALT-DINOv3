"""Train SALT on a single dataset.

Example:
  python train.py --train_root /data/MAS3K_RMAS_COD --size 336 --bs 32 \
      --epochs 20 --save runs/salt_mas336

The training set directory must contain RGB/ and GT/ subfolders. The saved
checkpoint holds only the trainable parameters (LoRA + neck + decoder + side-stem).
"""
import os
import time
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from salt import SALT, CODLoss, FullDataset


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True, help="dir with RGB/ and GT/")
    ap.add_argument("--save", required=True, help="output dir for the checkpoint")
    ap.add_argument("--backbone", default="dinov3", choices=["dinov3", "dinov2"])
    ap.add_argument("--size", type=int, default=336)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--no_sidestem", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1024)
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.save, exist_ok=True)

    ds = FullDataset(args.train_root, args.size, mode='train')
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, prefetch_factor=8 if args.workers else None)

    model = SALT(img_size=args.size, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                 use_sidestem=not args.no_sidestem, backbone=args.backbone).cuda()
    tp = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in tp); n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_tr:,} / {n_all:,} ({100 * n_tr / n_all:.3f}%)", flush=True)

    # Gradient accumulation keeps the effective batch >= 16 for small --bs.
    accum = max(1, 16 // args.bs) if args.bs < 8 else 1
    optimizer = torch.optim.AdamW(tp, lr=args.lr, weight_decay=args.weight_decay)
    total_iters = max(1, (len(loader) * args.epochs) // accum)
    sched = CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-7)
    criterion = CODLoss(weight_structure=1.0)

    model.train()
    best = float('inf'); t0 = time.time()
    for ep in range(args.epochs):
        eloss = 0.0
        for i, batch in enumerate(loader):
            rgb = batch['image'].cuda(); target = batch['label'].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_aux, out_main = model(rgb)
            loss = criterion(out_aux, target) + criterion(out_main, target)
            eloss += loss.item()
            (loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(tp, 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); sched.step()
            if i % 50 == 0:
                print(f"ep{ep + 1}-{i + 1}: loss {loss.item():.4f} "
                      f"lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)
        eloss /= len(loader)
        print(f"[ep {ep + 1}/{args.epochs}] loss {eloss:.4f}  elapsed {time.time() - t0:.0f}s", flush=True)
        if eloss < best:
            best = eloss
            torch.save(model.trainable_state_dict(), os.path.join(args.save, "salt-best.pth"))
            print(f"  saved (best loss {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
