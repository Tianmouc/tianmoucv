import torch
import torch.nn.functional as F


def local_norm(Diff: torch.Tensor) -> torch.Tensor:
    """Normalize a two-channel gradient field by its largest magnitude."""
    grad_norm = torch.hypot(Diff[0], Diff[1]).amax().clamp_min(1e-9)
    return Diff / grad_norm


def _prepare_derivatives(SD: torch.Tensor, TD: torch.Tensor):
    if not isinstance(SD, torch.Tensor) or not isinstance(TD, torch.Tensor):
        raise TypeError("SD and TD must be torch tensors")
    if SD.ndim != 3 or SD.shape[0] != 2:
        raise ValueError("SD must have shape [2, height, width]")
    if TD.ndim != 3 or TD.shape != (1, *SD.shape[-2:]):
        raise ValueError("TD must have shape [1, height, width] matching SD")
    dtype = torch.float64 if SD.dtype == torch.float64 else torch.float32
    spatial = SD.to(dtype=dtype)
    temporal = TD.to(device=spatial.device, dtype=dtype)
    return spatial, temporal[0]


def _resize_flow(flow: torch.Tensor, size):
    height, width = flow.shape[-2:]
    target_height, target_width = size
    if (height, width) == size:
        return flow
    resized = F.interpolate(flow.unsqueeze(0), size=size, mode="bilinear",
                            align_corners=False).squeeze(0)
    scale = flow.new_tensor((target_width / width, target_height / height))
    return resized * scale[:, None, None]


def _output_flow(flow: torch.Tensor, ifInterploted: bool):
    if not ifInterploted:
        return flow
    height, width = flow.shape[-2:]
    return _resize_flow(flow, (height * 2, width * 2))


def _box_sum(values: torch.Tensor, win: int, stride: int):
    pooled = F.avg_pool2d(values[None, None], kernel_size=win, stride=stride,
                          padding=win // 2, count_include_pad=True)
    return pooled[0, 0] * (win * win)


def _interpolate_sampled_flow(flow: torch.Tensor, size, stride: int):
    if stride == 1:
        return flow
    height, width = size
    sampled_height, sampled_width = flow.shape[-2:]
    rows = torch.arange(height, device=flow.device, dtype=flow.dtype) / stride
    columns = torch.arange(width, device=flow.device, dtype=flow.dtype) / stride
    rows = 2 * rows / max(sampled_height - 1, 1) - 1
    columns = 2 * columns / max(sampled_width - 1, 1) - 1
    grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)[None]
    return F.grid_sample(flow[None], grid, mode="bilinear",
                         padding_mode="border", align_corners=True)[0]


def LK_optical_flow(SD: torch.Tensor, TD: torch.Tensor, win=5,
                    stride=0, mask=None, ifInterploted=False) -> torch.Tensor:
    """Estimate (u, v) from Ix*u + Iy*v + It = 0 using windowed least squares.

    SD and TD use the same [height, width] grid; TD is the signed temporal
    derivative, not its negative. The default output is in that grid's pixel
    units. With ifInterploted=True, output and displacements are doubled to
    the corresponding RGB/COP [2*height, 2*width] grid.
    """
    spatial, temporal = _prepare_derivatives(SD, TD)
    height, width = spatial.shape[-2:]
    if not isinstance(win, int) or win < 3 or win % 2 == 0 or win > min(height, width):
        raise ValueError("win must be odd, at least 3, and fit within SD")
    if not isinstance(stride, int) or stride < 0:
        raise ValueError("stride must be a non-negative integer")
    stride = max(1, win // 2) if stride == 0 else stride

    weights = torch.ones_like(temporal)
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.shape != (height, width):
            raise ValueError("mask must have shape [height, width]")
        weights = (mask.to(device=spatial.device) != 0).to(spatial.dtype)

    grad_x, grad_y = spatial[0], spatial[1]
    sum_xx = _box_sum(weights * grad_x * grad_x, win, stride)
    sum_xy = _box_sum(weights * grad_x * grad_y, win, stride)
    sum_yy = _box_sum(weights * grad_y * grad_y, win, stride)
    rhs_x = -_box_sum(weights * grad_x * temporal, win, stride)
    rhs_y = -_box_sum(weights * grad_y * temporal, win, stride)
    determinant = sum_xx * sum_yy - sum_xy.square()
    trace = sum_xx + sum_yy
    valid = determinant > 1e-6 * trace.square()
    if mask is not None:
        valid &= _box_sum(weights, win, stride) >= min(5, win * win)
    safe_determinant = torch.where(valid, determinant, torch.ones_like(determinant))
    horizontal = torch.where(
        valid, (sum_yy * rhs_x - sum_xy * rhs_y) / safe_determinant,
        torch.zeros_like(determinant),
    )
    vertical = torch.where(
        valid, (sum_xx * rhs_y - sum_xy * rhs_x) / safe_determinant,
        torch.zeros_like(determinant),
    )
    flow = torch.stack((horizontal, vertical))
    flow = _interpolate_sampled_flow(flow, (height, width), stride)
    return _output_flow(flow, ifInterploted)


def _neighbor_average(values: torch.Tensor):
    padded = F.pad(values[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    return (padded[1:-1, :-2] + padded[1:-1, 2:]
            + padded[:-2, 1:-1] + padded[2:, 1:-1]) * 0.25


def HS_optical_flow(SD: torch.Tensor, TD: torch.Tensor, ifInterploted=False,
                    epsilon=1e-8, maxIteration=50, scales=4, labmda=10) -> torch.Tensor:
    """Estimate smooth flow with coarse-to-fine Horn-Schunck iterations.

    TD must contain the signed temporal derivative It. The output convention
    matches LK_optical_flow; labmda is the positive smoothness weight.
    """
    spatial, temporal = _prepare_derivatives(SD, TD)
    if not isinstance(maxIteration, int) or maxIteration < 1:
        raise ValueError("maxIteration must be a positive integer")
    if not isinstance(scales, int) or scales < 1:
        raise ValueError("scales must be a positive integer")
    if labmda <= 0 or epsilon < 0:
        raise ValueError("labmda must be positive and epsilon non-negative")

    height, width = spatial.shape[-2:]
    pyramid_sizes = []
    for level in reversed(range(scales)):
        size = (max(1, round(height / 2 ** level)),
                max(1, round(width / 2 ** level)))
        if size not in pyramid_sizes:
            pyramid_sizes.append(size)

    flow = None
    for level_height, level_width in pyramid_sizes:
        size = (level_height, level_width)
        derivatives = F.interpolate(spatial[None], size=size, mode="bilinear",
                                    align_corners=False)[0]
        grad_x = derivatives[0] * (width / level_width)
        grad_y = derivatives[1] * (height / level_height)
        time_derivative = F.interpolate(
            temporal[None, None], size=size, mode="bilinear",
            align_corners=False,
        )[0, 0]
        flow = (spatial.new_zeros((2, *size)) if flow is None
                else _resize_flow(flow, size))
        denominator = labmda ** 2 + grad_x.square() + grad_y.square()
        for _ in range(maxIteration):
            mean_horizontal = _neighbor_average(flow[0])
            mean_vertical = _neighbor_average(flow[1])
            residual = (grad_x * mean_horizontal + grad_y * mean_vertical
                        + time_derivative) / denominator
            updated = torch.stack((
                mean_horizontal - grad_x * residual,
                mean_vertical - grad_y * residual,
            ))
            change = (updated - flow).abs().amax()
            flow = updated
            if change <= epsilon:
                break
    return _output_flow(flow, ifInterploted)
