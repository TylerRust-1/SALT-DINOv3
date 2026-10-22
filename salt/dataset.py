"""RGB datasets for SALT. A dataset directory must contain `RGB/` (input images)
and `GT/` (binary ground-truth masks) with matching, naturally-sortable filenames.
Images are square-resized (stretched); the distortion acts as implicit augmentation
and outperformed aspect-preserving padding in our experiments."""
import os
import re
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def _list(root):
    return sorted([os.path.join(root, f) for f in os.listdir(root)
                   if f.lower().endswith(('.jpg', '.png'))], key=natural_sort_key)


class ToTensor:
    def __call__(self, d):
        d['image'] = TF.to_tensor(d['image'])
        d['label'] = TF.to_tensor(d['label'])
        return d


class Resize:
    """Square resize (stretch) to (size, size)."""
    def __init__(self, size):
        self.size = size

    def __call__(self, d):
        d['image'] = d['image'].resize((self.size, self.size), resample=Image.BILINEAR)
        d['label'] = d['label'].resize((self.size, self.size), resample=Image.NEAREST)
        return d


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, d):
        if random.random() < self.p:
            d['image'] = TF.hflip(d['image']); d['label'] = TF.hflip(d['label'])
        return d


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, d):
        if random.random() < self.p:
            d['image'] = TF.vflip(d['image']); d['label'] = TF.vflip(d['label'])
        return d


class RandomGamma:
    """Log-uniform random gamma on the image (label unchanged)."""
    def __init__(self, low=0.5, high=2.0, p=0.5):
        self.log_low, self.log_high, self.p = float(np.log(low)), float(np.log(high)), p

    def __call__(self, d):
        if random.random() < self.p:
            gamma = float(np.exp(random.uniform(self.log_low, self.log_high)))
            d['image'] = TF.adjust_gamma(d['image'], gamma)
        return d


class Normalize:
    """Normalize RGB to mean=std=0.5 (the model undoes this and re-applies
    ImageNet stats for the backbone input)."""
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean, self.std = mean, std

    def __call__(self, d):
        d['image'] = TF.normalize(d['image'], self.mean, self.std)
        return d


class FullDataset(Dataset):
    """Training/validation dataset. `root` must hold `RGB/` and `GT/`."""
    def __init__(self, root, size, mode='train'):
        self.images = _list(os.path.join(root, "RGB"))
        self.gts = _list(os.path.join(root, "GT"))
        assert len(self.images) == len(self.gts), "image / gt count mismatch"
        aug = [RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5), RandomGamma(0.5, 2.0, 0.5)] \
            if mode == 'train' else []
        self.transform = transforms.Compose([Resize(size), *aug, ToTensor(), Normalize()])

    def __getitem__(self, i):
        d = {'image': Image.open(self.images[i]).convert('RGB'),
             'label': Image.open(self.gts[i]).convert('L')}
        return self.transform(d)

    def __len__(self):
        return len(self.images)


class TestDataset:
    """Evaluation loader: yields (image_tensor[1,3,H,W], gt_array[H0,W0], name).
    The GT is returned at its native resolution; resize predictions back to it."""
    def __init__(self, root, size):
        self.images = _list(os.path.join(root, "RGB"))
        self.gts = _list(os.path.join(root, "GT"))
        self.length = len(self.images)
        self.index = 0
        self.transform = transforms.Compose([Resize(size), ToTensor(), Normalize()])

    def load_data(self):
        image = Image.open(self.images[self.index]).convert('RGB')
        gt = np.array(Image.open(self.gts[self.index]).convert('L'))
        d = {'image': image,
             'label': Image.fromarray(np.zeros((image.size[1], image.size[0]), np.uint8), mode='L')}
        d = self.transform(d)
        name = os.path.basename(self.images[self.index])
        self.index += 1
        return d['image'].unsqueeze(0), gt, name
