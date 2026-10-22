"""SALT: Side-stem, Attention-gated U-Net, Low-rank Tuning — frozen-backbone
adaptation for camouflaged, marine, and salient object detection."""
from .model import SALT, REPOS
from .loss import CODLoss, structure_loss
from .dataset import FullDataset, TestDataset

__all__ = ["SALT", "REPOS", "CODLoss", "structure_loss", "FullDataset", "TestDataset"]
