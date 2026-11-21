import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


def _gradients(Ix, Iy):
    grad_x = torch.as_tensor(Ix)
    grad_y = torch.as_tensor(Iy, device=grad_x.device)
    if grad_x.ndim != 2 or grad_x.shape != grad_y.shape:
        raise ValueError("Ix and Iy must be two-dimensional gradients of equal shape")
    if not grad_x.is_floating_point():
        grad_x = grad_x.float()
    grad_y = grad_y.to(dtype=grad_x.dtype)
    return grad_x, grad_y


def sobel_operator(Ix, Iy):
    grad_x, grad_y = _gradients(Ix, Iy)
    return torch.hypot(grad_x, grad_y), torch.atan2(grad_y, grad_x)


def gaussain_kernel(size=5, sigma=2, device=None, dtype=None):
    size = int(size)
    if size < 1 or sigma <= 0:
        raise ValueError("size and sigma must be positive")
    if size % 2 == 0:
        size += 1
    coordinates = torch.arange(size, device=device, dtype=dtype or torch.float32) - size // 2
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    kernel = torch.exp(-(grid_x.square() + grid_y.square()) / (2 * sigma * sigma))
    return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)


def gaussian_smooth(inputTensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if inputTensor.ndim != 4:
        raise ValueError("inputTensor must have shape [batch, channels, height, width]")
    if not inputTensor.is_floating_point():
        inputTensor = inputTensor.float()
    kernel = kernel.to(device=inputTensor.device, dtype=inputTensor.dtype)
    padding = kernel.shape[-1] // 2
    padding_mode = "reflect" if min(inputTensor.shape[-2:]) > padding else "replicate"
    padded = F.pad(inputTensor, (padding, padding, padding, padding), mode=padding_mode)
    channels = inputTensor.shape[1]
    return F.conv2d(padded, kernel.expand(channels, 1, -1, -1), groups=channels)


def _second_moments(grad_x, grad_y, size, sigma):
    kernel = gaussain_kernel(size, sigma, device=grad_x.device, dtype=grad_x.dtype)
    products = torch.stack((grad_x.square(), grad_y.square(), grad_x * grad_y), dim=0)
    smoothed = gaussian_smooth(products.unsqueeze(0), kernel).squeeze(0)
    return smoothed[0], smoothed[1], smoothed[2]


def _corner_peaks(response, nms_size, threshold=0, maximum=None):
    if nms_size < 1 or nms_size % 2 == 0:
        raise ValueError("nmsSize must be a positive odd integer")
    if maximum == 0:
        return []
    pooled = F.max_pool2d(response[None, None], nms_size, stride=1, padding=nms_size // 2)[0, 0]
    peak_mask = (response > threshold) & (response == pooled)
    coordinates = torch.nonzero(peak_mask)
    if maximum is not None and coordinates.shape[0] > maximum:
        strengths = response[coordinates[:, 0], coordinates[:, 1]]
        coordinates = coordinates[torch.topk(strengths, maximum).indices]
    return [tuple(point) for point in coordinates.cpu().tolist()]


def HarrisCorner(Ix: torch.Tensor, Iy: torch.Tensor, thresh=0.1, k=0.1,
                 size=5, sigma=1, nmsSize=11):
    """Return (row, column) Harris corners; thresh is relative to the positive peak."""
    if not 0 <= thresh <= 1:
        raise ValueError("thresh must be between 0 and 1")
    grad_x, grad_y = _gradients(Ix, Iy)
    moment_xx, moment_yy, moment_xy = _second_moments(grad_x, grad_y, size, sigma)
    trace = moment_xx + moment_yy
    response = moment_xx * moment_yy - moment_xy.square() - k * trace.square()
    positive_peak = response.max()
    if positive_peak <= 0:
        return []
    return _corner_peaks(response, nmsSize, positive_peak * thresh)


def TomasiCorner(Ix: torch.Tensor, Iy: torch.Tensor, index=1000,
                 size=5, sigma=2, nmsSize=11):
    """Return up to index strongest (row, column) Shi-Tomasi corners."""
    if index < 0:
        raise ValueError("index must be non-negative")
    grad_x, grad_y = _gradients(Ix, Iy)
    moment_xx, moment_yy, moment_xy = _second_moments(grad_x, grad_y, size, sigma)
    discriminant = (moment_xx - moment_yy).square() + 4 * moment_xy.square()
    response = (moment_xx + moment_yy - torch.sqrt(discriminant.clamp_min(0))) / 2
    return _corner_peaks(response, nmsSize, maximum=index)


def _histogram_descriptor(magnitudes, angles, cells_per_axis, orientation_bins, angle_range):
    patch_size = magnitudes.shape[0]
    cell_size = patch_size // cells_per_axis
    cell_rows = torch.arange(patch_size, device=magnitudes.device) // cell_size
    cell_columns = torch.arange(patch_size, device=magnitudes.device) // cell_size
    cell_indices = (cell_rows[:, None] * cells_per_axis + cell_columns[None, :]) * orientation_bins

    bin_positions = torch.remainder(angles, angle_range) * (orientation_bins / angle_range)
    lower_bins = torch.floor(bin_positions).long() % orientation_bins
    upper_bins = (lower_bins + 1) % orientation_bins
    upper_weights = bin_positions - torch.floor(bin_positions)

    descriptor = torch.zeros(cells_per_axis * cells_per_axis * orientation_bins,
                             device=magnitudes.device, dtype=magnitudes.dtype)
    descriptor.scatter_add_(0, (cell_indices + lower_bins).flatten(),
                            (magnitudes * (1 - upper_weights)).flatten())
    descriptor.scatter_add_(0, (cell_indices + upper_bins).flatten(),
                            (magnitudes * upper_weights).flatten())
    descriptor = F.normalize(descriptor, dim=0, eps=1e-8)
    descriptor = descriptor.clamp(max=0.2)
    return F.normalize(descriptor, dim=0, eps=1e-8)


def hog(Ix: torch.Tensor, Iy: torch.Tensor, kplist: list):
    """Return valid (x, y) points and 36-dimensional local HOG descriptors."""
    grad_x, grad_y = _gradients(Ix, Iy)
    magnitudes = torch.hypot(grad_x, grad_y)
    angles = torch.atan2(grad_y, grad_x)
    valid_points = []
    descriptors = []
    height, width = grad_x.shape
    for point in kplist:
        column, row = (int(round(float(coordinate))) for coordinate in point[:2])
        if row < 8 or column < 8 or row + 8 > height or column + 8 > width:
            continue
        patch_magnitudes = magnitudes[row - 8:row + 8, column - 8:column + 8]
        patch_angles = angles[row - 8:row + 8, column - 8:column + 8]
        descriptors.append(_histogram_descriptor(patch_magnitudes, patch_angles, 2, 9, math.pi))
        valid_points.append((column, row))
    return valid_points, descriptors


def _gradient_pyramid(grad_x, grad_y, num_levels):
    for level in range(num_levels):
        yield grad_x, grad_y
        height, width = grad_x.shape
        if min(height, width) < 32:
            break
        next_shape = (height // 2, width // 2)
        grad_x = F.interpolate(grad_x[None, None], size=next_shape,
                               mode="bilinear", align_corners=False)[0, 0]
        grad_y = F.interpolate(grad_y[None, None], size=next_shape,
                               mode="bilinear", align_corners=False)[0, 0]


def steadyHarrisCornerForSIFT(Ix: torch.Tensor, Iy: torch.Tensor, num_levels=6,
                             thresh=0.1, k=0.1, size=5, sigma=1, nmsSize=11):
    """Return (x, y, one-based level) corners supported at adjacent scales."""
    if num_levels < 1:
        raise ValueError("num_levels must be positive")
    grad_x, grad_y = _gradients(Ix, Iy)
    levels = list(_gradient_pyramid(grad_x, grad_y, num_levels))
    points_by_level = []
    for level_x, level_y in levels:
        corners = HarrisCorner(level_x, level_y, thresh=thresh, k=k,
                               size=size, sigma=sigma, nmsSize=nmsSize)
        points_by_level.append(np.asarray([(column, row) for row, column in corners],
                                          dtype=np.float32).reshape(-1, 2))

    stable_points = []
    for level_index, points in enumerate(points_by_level):
        if len(points) == 0:
            continue
        if len(points_by_level) == 1:
            supported = np.ones(len(points), dtype=bool)
        else:
            supported = np.zeros(len(points), dtype=bool)
            original_points = points * (2 ** level_index)
            for neighbor_index in (level_index - 1, level_index + 1):
                if neighbor_index < 0 or neighbor_index >= len(points_by_level):
                    continue
                neighbor_points = points_by_level[neighbor_index]
                if len(neighbor_points) == 0:
                    continue
                neighbor_tree = cKDTree(neighbor_points * (2 ** neighbor_index))
                radius = 2 * (2 ** max(level_index, neighbor_index))
                distances, _ = neighbor_tree.query(original_points, distance_upper_bound=radius)
                supported |= np.isfinite(distances)
        for column, row in points[supported]:
            stable_points.append((int(column), int(row), level_index + 1))
    return stable_points


def compute_multiscale_sift_descriptor(Ix: torch.Tensor, Iy: torch.Tensor,
                                       kp_list: list, num_levels=6):
    """Return 128-dimensional SIFT-style descriptors and original-scale (x, y) points."""
    if num_levels < 1:
        raise ValueError("num_levels must be positive")
    grad_x, grad_y = _gradients(Ix, Iy)
    pyramid = list(_gradient_pyramid(grad_x, grad_y, num_levels))
    gradient_maps = [(torch.hypot(level_x, level_y), torch.atan2(level_y, level_x))
                     for level_x, level_y in pyramid]

    coordinates = torch.arange(-8, 8, device=grad_x.device, dtype=grad_x.dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    gaussian_weights = torch.exp(-(grid_x.square() + grid_y.square()) / (2 * 8 ** 2))

    descriptors = []
    valid_points = []
    for column, row, level in kp_list:
        level_index = int(level) - 1
        if level_index < 0 or level_index >= len(gradient_maps):
            continue
        column = int(round(float(column)))
        row = int(round(float(row)))
        magnitudes, angles = gradient_maps[level_index]
        height, width = magnitudes.shape
        if row < 8 or column < 8 or row + 8 > height or column + 8 > width:
            continue
        patch_magnitudes = magnitudes[row - 8:row + 8, column - 8:column + 8] * gaussian_weights
        patch_angles = angles[row - 8:row + 8, column - 8:column + 8]
        orientation_histogram = torch.zeros(36, device=grad_x.device, dtype=grad_x.dtype)
        orientation_indices = torch.floor(torch.remainder(patch_angles, 2 * math.pi)
                                          * (36 / (2 * math.pi))).long() % 36
        orientation_histogram.scatter_add_(0, orientation_indices.flatten(),
                                           patch_magnitudes.flatten())
        dominant_bin = torch.argmax(orientation_histogram)
        dominant_angle = (dominant_bin + 0.5) * (2 * math.pi / 36)
        relative_angles = patch_angles - dominant_angle
        descriptors.append(_histogram_descriptor(patch_magnitudes, relative_angles,
                                                  4, 8, 2 * math.pi))
        scale = 2 ** level_index
        valid_points.append((column * scale, row * scale))
    return descriptors, valid_points
