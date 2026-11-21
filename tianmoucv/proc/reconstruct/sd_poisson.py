"""SD grayscale reconstruction on the native sensor grid (Pro GUI convention)."""
from functools import lru_cache
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def sd_gradients(sd):
    """Map raw SDL/SDR [2,H,W] to native-grid horizontal and vertical gradients."""
    sd = torch.as_tensor(sd)
    if sd.ndim != 3 or sd.shape[0] != 2 or min(sd.shape[1:]) < 1:
        raise ValueError("expected SDL/SDR with shape [2,H,W]")
    sd = torch.nan_to_num(sd.float()).clamp(-1e6, 1e6)
    even = sd[:, ::2]
    odd = sd[:, 1::2]
    if sd.shape[1] % 2:
        odd = torch.cat((odd, even[:, -1:]), dim=1)
    ul, ur = even.unbind()
    dl, dr = odd.unbind()
    gx = (ul + dl - ur - dr) / math.sqrt(8)
    gy = (ul - dl + ur - dr) / math.sqrt(8)
    return tuple(v.repeat_interleave(2, dim=0)[:sd.shape[1]] for v in (gx, gy))


@lru_cache(maxsize=8)
def _eigenvalues(h, w):
    x = np.arange(w) * (np.pi / w)
    y = np.arange(h) * (np.pi / h)
    values = 4 - 2 * np.cos(x) - 2 * np.cos(y[:, None])
    values[0, 0] = 1
    return values


def poisson_gray(gx, gy):
    """Integrate forward edge gradients with natural boundaries and zero DC.

    Odd dimensions follow the GUI's replicate-pad, solve, crop convention.
    CPU inference uses OpenCV DCT. Even reflection gives the same Neumann
    solution with torch.fft for accelerated tensors or autograd.
    """
    gx, gy = torch.as_tensor(gx), torch.as_tensor(gy)
    if gx.ndim != 2 or gx.shape != gy.shape or min(gx.shape) < 1:
        raise ValueError("gx and gy must be nonempty 2-D planes of equal shape")
    dtype = torch.float64 if gx.dtype == torch.float64 else torch.float32
    gx, gy = gx.to(dtype), gy.to(device=gx.device, dtype=dtype)
    h, w = gx.shape
    if h % 2 or w % 2:
        padded = F.pad(torch.stack((gx, gy))[None], (0, w % 2, 0, h % 2), mode='replicate')
        gx, gy = padded[0].unbind()
    rhs = torch.zeros_like(gx)
    rhs[:, :-1] -= gx[:, :-1]
    rhs[:, 1:] += gx[:, :-1]
    rhs[:-1] -= gy[:-1]
    rhs[1:] += gy[:-1]
    height, width = rhs.shape
    if rhs.device.type == 'cpu' and not rhs.requires_grad:
        spectrum = cv2.dct(rhs.numpy())
        spectrum /= _eigenvalues(height, width)
        spectrum[0, 0] = 0
        return torch.from_numpy(cv2.idct(spectrum)[:h, :w].copy())

    extended = torch.cat((rhs, rhs.flip((1,))), dim=1)
    extended = torch.cat((extended, extended.flip((0,))), dim=0)
    x = torch.arange(width + 1, device=gx.device, dtype=dtype)
    y = torch.arange(2 * height, device=gx.device, dtype=dtype)
    eigen = 4 * (torch.sin(math.pi * x / (2 * width)).square() +
                 torch.sin(math.pi * y[:, None] / (2 * height)).square())
    eigen[0, 0] = 1
    spectrum = torch.fft.rfft2(extended) / eigen
    spectrum[0, 0] = 0
    return torch.fft.irfft2(spectrum, s=extended.shape)[:h, :w]


def sd_gray_preview(gray):
    """Pro GUI grayscale display: 1% tails, 8-bit rounding, then linear 2x width."""
    array = gray.detach().cpu().numpy()
    k = array.size // 100
    hi_index = array.size - 1 - k
    ordered = np.partition(array.ravel(), (k, hi_index))
    lo, hi = ordered[k], ordered[hi_index]
    if hi - lo < 1e-6:
        pixels = np.full(array.shape, 127, dtype=np.uint8)
    else:
        pixels = np.rint((array - lo) * (255. / (hi - lo))).clip(0, 255).astype(np.uint8)
    return cv2.resize(pixels, (2 * array.shape[1], array.shape[0]), interpolation=cv2.INTER_LINEAR)
