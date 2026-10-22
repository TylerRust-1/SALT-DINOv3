"""SALT: Side-stem, Attention-gated U-Net, Low-rank Tuning.

A lightweight adaptation of a frozen isotropic ViT backbone (DINOv3 ViT-L/16 by
default, DINOv2 ViT-L/14 optional) for binary foreground segmentation
(camouflaged, marine, and salient object detection):

  frozen backbone + flat rank-r LoRA on the attention projections
  + a ViTDet-style simple feature pyramid neck
  + an attention-gated U-Net decoder
  + a full-resolution CNN side-stem for high-frequency detail
  + a structure (weighted BCE + weighted IoU) training loss.

Only the LoRA factors, the pyramid neck, the decoder, and the side-stem train
(~5.6M parameters, under 2% of the model for DINOv3 ViT-L); the backbone stays
frozen. The backbone
is isotropic: it has no inter-stage pooling, so the patch-token grid grows with
input resolution and is preserved through every block. That token geometry is
what lets the model turn higher input resolution into accuracy.

This is the public method release. The mechanism-study scaffolding from the
research repository (pooling / windowed-attention ablations, sparse high-res
refinement, mixed-resolution token sets) is intentionally omitted here for
clarity; this file is the clean training/inference model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft.tuners.lora import Linear as LoraLinear

# Ungated transformers-format backbones.
REPOS = {
    "dinov3": "camenduru/dinov3-vitl16-pretrain-lvd1689m",  # ViT-L/16, SSL LVD-1689M
    "dinov2": "facebook/dinov2-large",                       # ViT-L/14, SSL LVD-142M
}
# Both backbones expect ImageNet normalization; the dataset feeds mean=std=0.5,
# so we undo that and re-apply ImageNet stats for the backbone input only.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Decoder building blocks
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al.): the upsampled decoder feature gates
    the encoder skip, suppressing background activations before concatenation."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.gelu = nn.GELU()

    def forward(self, g, x):
        g1 = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)
        g1 = self.W_g(g1)
        x1 = self.W_x(x)
        psi = self.psi(self.gelu(g1 + x1))
        return (0.5 + 0.5 * psi) * x


class AttentionUp(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.attn_gate = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=in_channels // 2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x_decoder, x_skip):
        x_up = self.up(x_decoder)
        x_skip_clean = self.attn_gate(g=x_up, x=x_skip)
        return self.conv(torch.cat([x_skip_clean, x_up], dim=1))


class SideStem(nn.Module):
    """Multi-stride CNN on the raw full-resolution input, producing features at
    strides 1, 2, 4 that are fused into the decoder to recover high-frequency
    boundary detail the coarse-patch ViT cannot represent."""
    def __init__(self, ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ch, ch, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(ch), nn.GELU())
        self.down4 = nn.Sequential(
            nn.Conv2d(ch, ch, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(ch), nn.GELU())

    def forward(self, x):
        f1 = self.stem(x)    # stride 1 (full-res)
        f2 = self.down2(f1)  # stride 2
        f4 = self.down4(f2)  # stride 4
        return f1, f2, f4


class DINOPyramid(nn.Module):
    """ViTDet-style simple feature pyramid: from the single stride-P token grid,
    produce three levels (fine/mid/coarse) at `out_ch` channels for the decoder.
    The decoder upsamples by exactly 2x per stage, so the levels sit at a strict
    1:2:4 size ratio set by interpolation (works for any grid G, including odd
    grids such as 21 at 336/16)."""
    def __init__(self, embed_dim=1024, out_ch=256):
        super().__init__()
        self.fine = nn.Conv2d(embed_dim, out_ch, kernel_size=3, padding=1)
        self.mid = nn.Conv2d(embed_dim, out_ch, kernel_size=1)
        self.coarse = nn.Conv2d(embed_dim, out_ch, kernel_size=1)

    def forward(self, feat):  # feat: [B, C, G, G]
        G = feat.shape[-1]
        sc = max(1, G // 2)
        coarse = self.coarse(F.interpolate(feat, size=(sc, sc), mode="bilinear", align_corners=False))
        mid = self.mid(F.interpolate(feat, size=(2 * sc, 2 * sc), mode="bilinear", align_corners=False))
        fine = self.fine(F.interpolate(feat, size=(4 * sc, 4 * sc), mode="bilinear", align_corners=False))
        return fine, mid, coarse


# ---------------------------------------------------------------------------
# LoRA injection
# ---------------------------------------------------------------------------
def _find_blocks(model):
    """Return the transformer block ModuleList, whichever attribute path the HF
    model uses (DINOv3: model.layer; DINOv2: model.encoder.layer)."""
    best = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 12 and hasattr(mod[0], "attention"):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("could not locate transformer block list")
    return best


def _apply_lora_dino(model, r, alpha, dropout=0.05):
    """Inject flat-rank LoRA into every attention-projection Linear of every block,
    handling both DINOv2 (attention.attention.{query,key,value}, attention.output.dense)
    and DINOv3 (attention.{q,k,v,o}_proj)."""
    blocks = _find_blocks(model)
    n = 0
    for blk in blocks:
        attn = blk.attention
        for _, parent in list(attn.named_modules()):
            for child_name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, child_name, LoraLinear(
                        child, adapter_name="default", r=r, lora_alpha=alpha,
                        lora_dropout=dropout, fan_in_fan_out=False, bias="none",
                        use_rslora=False))
                    n += 1
    print(f"Injected LoRA (r={r}, alpha={alpha}) into {n} attention projections "
          f"({len(blocks)} blocks).")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SALT(nn.Module):
    """SALT segmentation model. Returns (aux, main) logits at the input resolution;
    use `main` for the final mask. The backbone runs at `img_size` (square); inputs
    of any size are resized to it for the backbone and the outputs are interpolated
    back to the original resolution."""
    def __init__(self, img_size=336, lora_r=8, lora_alpha=16, use_sidestem=True,
                 backbone="dinov3", repo=None):
        super().__init__()
        self.img_size = img_size
        self.backbone_name = backbone

        repo = repo or REPOS[backbone]
        self.backbone = AutoModel.from_pretrained(repo)
        cfg = self.backbone.config
        self.embed_dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.lora_enabled = lora_r is not None and lora_r > 0
        if self.lora_enabled:
            _apply_lora_dino(self.backbone, lora_r, lora_alpha)

        # Constant normalization stats; non-persistent so checkpoints stay minimal.
        self.register_buffer("imnet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("imnet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
                             persistent=False)

        # Backbone-specific neck.
        self.pyramid = DINOPyramid(self.embed_dim, out_ch=256)

        # Attention-gated U-Net decoder.
        ch = 64
        self.ncm = nn.ModuleList([DoubleConv(256, ch), DoubleConv(256, ch), DoubleConv(256, ch)])
        self.up1 = AttentionUp(ch, ch, ch)
        self.up2 = AttentionUp(ch, ch, ch)
        self.head_aux = nn.Conv2d(ch, 1, 1)
        self.head_main = nn.Conv2d(ch, 1, 1)

        self.use_sidestem = use_sidestem
        if use_sidestem:
            side_ch = 32
            self.side_stem = SideStem(ch=side_ch)
            self.fuse_low = DoubleConv(ch + side_ch, ch)
            self.fuse_med = DoubleConv(ch + side_ch, ch)
            self.fuse_high = DoubleConv(ch + side_ch, ch)

    def _encode(self, x_lr):
        """Run the frozen + LoRA backbone and reshape patch tokens to [B, C, G, G]."""
        img01 = x_lr * 0.5 + 0.5  # undo the dataset's mean=std=0.5 normalization
        x_in = (img01 - self.imnet_mean) / self.imnet_std
        out = self.backbone(pixel_values=x_in)
        tok = out.last_hidden_state
        B, N, C = tok.shape
        G = x_lr.shape[-1] // self.patch_size            # patch grid for square input
        tok = tok[:, N - G * G:, :]                       # drop CLS + any register prefix
        return tok.transpose(1, 2).reshape(B, C, G, G)

    def forward(self, x):
        hr_size = x.shape[2:]
        if hr_size[0] != self.img_size or hr_size[1] != self.img_size:
            x_lr = F.interpolate(x, size=(self.img_size, self.img_size),
                                 mode='bilinear', align_corners=False)
        else:
            x_lr = x

        feat = self._encode(x_lr)
        s1, s2, s3 = self.pyramid(feat)           # fine, mid, coarse (256 ch)
        s1_lr = self.ncm[0](s1)
        s2_lr = self.ncm[1](s2)
        s3_lr = self.ncm[2](s3)

        x144 = self.up1(s3_lr, s2_lr)
        x288 = self.up2(x144, s1_lr)

        out_aux = self.head_aux(x144)
        out_aux = F.interpolate(out_aux, size=hr_size, mode="bilinear", align_corners=False)

        if self.use_sidestem:
            side1, side2, side4 = self.side_stem(x)
            side4_at = F.interpolate(side4, size=x288.shape[-2:], mode="bilinear", align_corners=False)
            x_low = self.fuse_low(torch.cat([x288, side4_at], dim=1))
            x_med = F.interpolate(x_low, size=side2.shape[-2:], mode="bilinear", align_corners=False)
            x_med = self.fuse_med(torch.cat([x_med, side2], dim=1))
            x_high = F.interpolate(x_med, size=side1.shape[-2:], mode="bilinear", align_corners=False)
            x_high = self.fuse_high(torch.cat([x_high, side1], dim=1))
            feat_final = x_high
        else:
            feat_final = x288

        out_main = self.head_main(feat_final)
        if out_main.shape[-2:] != hr_size:
            out_main = F.interpolate(out_main, size=hr_size, mode="bilinear", align_corners=False)
        return out_aux, out_main

    def trainable_state_dict(self):
        """State dict of only the trainable parameters (LoRA + neck + decoder +
        side-stem) — the lightweight checkpoint to save and ship."""
        keep = ("pyramid", "ncm", "up1", "up2", "head_", "side_stem", "fuse_")
        return {k: v for k, v in self.state_dict().items()
                if any(k.startswith(p) for p in keep) or "lora" in k.lower()}
