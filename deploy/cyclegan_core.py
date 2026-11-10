"""cyclegan_core.py – CycleGAN networks and helpers shared by the training notebook and the Flask app."""
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ------------------------------------------------------------------ generator
class ResnetBlock(nn.Module):
    """Residual block: (reflect-pad, 3x3 conv, instance-norm, ReLU) x2 with a skip connection."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """c7s1-64, d128, d256, R256 x n_blocks, u128, u64, c7s1-3 (Johnson et al. / CycleGAN paper)."""

    def __init__(self, ngf=64, n_blocks=9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        for i in range(2):                                   # down-sampling
            m = 2 ** i
            layers += [nn.Conv2d(ngf * m, ngf * m * 2, 3, 2, 1), nn.InstanceNorm2d(ngf * m * 2), nn.ReLU(True)]
        layers += [ResnetBlock(ngf * 4) for _ in range(n_blocks)]
        for i in range(2):                                   # up-sampling
            m = 2 ** (2 - i)
            layers += [nn.ConvTranspose2d(ngf * m, ngf * m // 2, 3, 2, 1, output_padding=1),
                       nn.InstanceNorm2d(ngf * m // 2), nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ------------------------------------------------------------------ discriminator
class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN: outputs a grid of real/fake scores, one per image patch."""

    def __init__(self, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        nf = 1
        for n in range(1, n_layers):
            prev, nf = nf, min(2 ** n, 8)
            layers += [nn.Conv2d(ndf * prev, ndf * nf, 4, 2, 1), nn.InstanceNorm2d(ndf * nf), nn.LeakyReLU(0.2, True)]
        prev, nf = nf, min(2 ** n_layers, 8)
        layers += [nn.Conv2d(ndf * prev, ndf * nf, 4, 1, 1), nn.InstanceNorm2d(ndf * nf), nn.LeakyReLU(0.2, True),
                   nn.Conv2d(ndf * nf, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(net, std=0.02):
    """Initialise conv weights from N(0, 0.02) as in the paper."""
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    net.apply(_init)
    return net


class ImagePool:
    """History buffer of previously generated images (stabilises discriminator training)."""

    def __init__(self, size=50):
        self.size, self.images = size, []

    def query(self, imgs):
        if self.size == 0:
            return imgs
        out = []
        for img in imgs:
            img = img.detach().unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img); out.append(img)
            elif random.random() > 0.5:                      # return an old image, store the new one
                i = random.randint(0, self.size - 1)
                out.append(self.images[i].clone()); self.images[i] = img
            else:
                out.append(img)
        return torch.cat(out)


# ------------------------------------------------------------------ image helpers
def pil_to_tensor(img, size):
    """PIL image -> normalised tensor (1,3,size,size) in [-1, 1]."""
    img = img.convert('RGB').resize((size, size), Image.BICUBIC)
    a = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (a * 2 - 1).unsqueeze(0)


def tensor_to_pil(t):
    """Tensor (1,3,H,W) or (3,H,W) in [-1, 1] -> PIL image."""
    t = t.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    a = ((t.clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(a)


# ------------------------------------------------------------------ sketch-vs-photo detector features
def image_features(img):
    """Hand-crafted features (colour, brightness, edge statistics) used to tell sketches from photos."""
    im = img.convert('RGB').resize((128, 128))
    a = np.asarray(im, dtype=np.float32) / 255.0
    sat = np.asarray(im.convert('HSV'), dtype=np.float32)[..., 1] / 255.0
    gray = a.mean(axis=2)
    colourful = (np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2]) + np.abs(a[..., 0] - a[..., 2])).mean()
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    hist = np.histogram(gray, bins=8, range=(0, 1))[0] / gray.size
    return np.array([sat.mean(), sat.std(), colourful, gray.mean(), gray.std(), (gray > 0.9).mean(), gx, gy, *hist],
                    dtype=np.float32)
