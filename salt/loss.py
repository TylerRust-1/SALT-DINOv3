"""Training losses for SALT. The default recipe uses the structure loss alone
(weighted BCE + weighted IoU); dice / SSIM / boundary terms are available and can
be enabled through CODLoss weights."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def structure_loss(pred, mask):
    """Weighted BCE + weighted IoU (F3Net-style). The weight upweights pixels near
    the object boundary (|avg_pool(mask) - mask|), so the loss focuses on edges.
    Uses a fixed (.mean()) normalization so large bodies do not dilute thin parts."""
    w = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (w * wbce).mean(dim=(2, 3))
    p = torch.sigmoid(pred)
    inter = ((p * mask) * w).sum(dim=(2, 3))
    union = ((p + mask) * w).sum(dim=(2, 3))
    wiou = 1 - (inter + 1e-6) / (union - inter + 1e-6)
    return (wbce + wiou).mean()


def dice_loss(pred, mask, eps=1e-6):
    p = torch.sigmoid(pred)
    inter = (p * mask).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def boundary_loss(pred, mask, kernel_size=3, eps=1e-6):
    """Edge-vs-edge BCE: a morphological-gradient boundary band is extracted from
    both sigmoid(pred) and GT, and matched pixel-for-pixel."""
    p = torch.sigmoid(pred)
    pad = kernel_size // 2

    def _edge(m):
        d = F.max_pool2d(m, kernel_size, stride=1, padding=pad)
        e = -F.max_pool2d(-m, kernel_size, stride=1, padding=pad)
        return (d - e).clamp(0, 1)

    eg, ep = _edge(mask), _edge(p).clamp(eps, 1 - eps)
    return -(eg * torch.log(ep) + (1 - eg) * torch.log(1 - ep)).mean()


def _gaussian_window(k, sigma, device, dtype):
    x = torch.arange(k, device=device, dtype=dtype) - k // 2
    g = torch.exp(-x.pow(2) / (2 * sigma * sigma)); g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_loss(pred, mask, window_size=11, sigma=1.5):
    """1 - local-window SSIM on sigmoid(pred) vs mask; punishes hazy boundaries."""
    p = torch.sigmoid(pred)
    pad = window_size // 2
    w = _gaussian_window(window_size, sigma, p.device, p.dtype)
    mu_p, mu_t = F.conv2d(p, w, padding=pad), F.conv2d(mask, w, padding=pad)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sig_p2 = F.conv2d(p * p, w, padding=pad) - mu_p2
    sig_t2 = F.conv2d(mask * mask, w, padding=pad) - mu_t2
    sig_pt = F.conv2d(p * mask, w, padding=pad) - mu_pt
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_pt + C1) * (2 * sig_pt + C2)) / ((mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2))
    return 1 - ssim.mean()


class CODLoss(nn.Module):
    """Weighted sum of structure / dice / SSIM / boundary terms. Defaults to the
    structure loss alone (the recipe used for the released results). Predictions
    are cast to float32 for numerical safety under autocast."""
    def __init__(self, weight_structure=1.0, weight_dice=0.0, weight_ssim=0.0, weight_boundary=0.0):
        super().__init__()
        self.ws, self.wd, self.wss, self.wb = (
            weight_structure, weight_dice, weight_ssim, weight_boundary)

    def _single(self, pred, mask):
        loss = self.ws * structure_loss(pred, mask)
        if self.wd > 0:
            loss = loss + self.wd * dice_loss(pred, mask)
        if self.wss > 0:
            loss = loss + self.wss * ssim_loss(pred, mask)
        if self.wb > 0:
            loss = loss + self.wb * boundary_loss(pred, mask)
        return loss

    def forward(self, preds, mask):
        if mask.max() > 1.0:
            mask = mask.float() / 255.0
        mask = (mask > 0.5).to(torch.float32)
        if isinstance(preds, torch.Tensor):
            return self._single(preds.to(torch.float32), mask)
        total = 0.0
        for pred in preds:
            p = pred.to(torch.float32)
            if p.shape[-2:] != mask.shape[-2:]:
                p = F.interpolate(p, size=mask.shape[-2:], mode='bilinear', align_corners=False)
            total = total + self._single(p, mask)
        return total
