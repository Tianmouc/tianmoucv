import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ===============================================================
# 描述子匹配与图像对齐
# ===============================================================
def _descriptor_array(descriptors):
    if descriptors is None:
        return np.empty((0, 0), dtype=np.float32)
    if torch.is_tensor(descriptors):
        descriptors = descriptors.detach().cpu().numpy()
    elif isinstance(descriptors, (list, tuple)) and descriptors and torch.is_tensor(descriptors[0]):
        descriptors = torch.stack(descriptors).detach().cpu().numpy()
    result = np.asarray(descriptors, dtype=np.float32)
    if result.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    if result.ndim != 2:
        raise ValueError("descriptors must have shape [count, dimensions]")
    return np.ascontiguousarray(result)


def feature_matching(des1: np.array, des2: np.array, ratio=0.85):
    """Return unique Lowe-ratio matches in the OpenCV knnMatch format."""
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")
    reference = _descriptor_array(des1)
    candidates = _descriptor_array(des2)
    if len(reference) == 0 or len(candidates) < 2:
        return []
    if reference.shape[1] != candidates.shape[1]:
        raise ValueError("descriptor dimensions must match")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(reference, candidates, k=2)
    accepted = [pair[0] for pair in pairs
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    used_candidates = set()
    unique_matches = []
    for match in sorted(accepted, key=lambda item: item.distance):
        if match.trainIdx in used_candidates:
            continue
        used_candidates.add(match.trainIdx)
        unique_matches.append([match])
    return unique_matches


def mini_l2_cost_matching(des1, des2, num=20):
    """Return the lowest-cost one-to-one descriptor pairs."""
    if num < 0:
        raise ValueError("num must be non-negative")
    reference = _descriptor_array(des1)
    candidates = _descriptor_array(des2)
    if len(reference) == 0 or len(candidates) == 0 or num == 0:
        return []
    if reference.shape[1] != candidates.shape[1]:
        raise ValueError("descriptor dimensions must match")
    costs = cdist(reference, candidates, metric="euclidean")
    reference_indices, candidate_indices = linear_sum_assignment(costs)
    pairs = sorted(zip(reference_indices, candidate_indices),
                   key=lambda pair: costs[pair[0], pair[1]])
    return [(int(reference_index), int(candidate_index))
            for reference_index, candidate_index in pairs[:num]]


def align_images(image, kpList1, kpList2, matches, canvas=None):
    """Estimate a homography from (x, y) point lists and return the warped image."""
    if len(matches) < 4:
        return image, None
    indices = []
    for item in matches:
        if isinstance(item, tuple) and len(item) == 2:
            indices.append(item)
        else:
            match = item[0] if isinstance(item, list) else item
            indices.append((match.queryIdx, match.trainIdx))
    source_points = np.float32([kpList1[reference_index] for reference_index, _ in indices])
    target_points = np.float32([kpList2[candidate_index] for _, candidate_index in indices])
    homography, _ = cv2.findHomography(source_points.reshape(-1, 1, 2),
                                       target_points.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
    if canvas is not None:
        image_width = image.shape[1]
        for source, target in zip(source_points, target_points):
            cv2.line(canvas, tuple(np.rint(source).astype(int)),
                     (int(round(target[0])) + image_width, int(round(target[1]))),
                     (255, 0, 0))
    if homography is None:
        return image, None
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, homography, (width, height)), homography


# ===============================================================
# SD 特征跟踪：预处理、双向 LK、补点与历史轨迹
# ===============================================================
def _normalize_sd_image(sd_img, previous_scale=None):
    """将 SD 幅值转成 LK 所需的灰度图，平滑尺度以减少逐帧对比度跳变。"""
    if torch.is_tensor(sd_img):
        sd_img = sd_img.detach().cpu().numpy()
    image = np.asarray(sd_img, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("sd_img must have shape [height, width]")
    magnitude = np.maximum(np.nan_to_num(image, nan=0, posinf=0, neginf=0), 0)
    positive = magnitude[magnitude > 0]
    current_scale = max(1.0, float(np.percentile(positive, 99))) if positive.size else 1.0
    # 使用上一帧的尺度作指数平滑，避免每帧独立归一化破坏 LK 的亮度一致性。
    scale = (current_scale if previous_scale is None
             else 0.9 * previous_scale + 0.1 * current_scale)
    gray = np.clip(magnitude * (255.0 / (1.5 * scale)), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(gray, (3, 3), 0.7), scale


class Feature_tracker_sd:
    """在 SD 幅值图上使用 Shi-Tomasi 选点和 OpenCV 金字塔 LK 跟踪。

    OpenCV 内部点坐标为 (x, y)，对外 history/trail_history 使用 (row, column)。
    history 只包含当前活跃轨迹；trail_history 保留近期轨迹供显示，不参与匹配。

    参数:
        max_features: 同时跟踪的最大特征点数。
        max_history: 每条轨迹最多保存的点数。
        max_displacement: 相邻 SD 帧允许的最大位移，单位为 SD 像素。
        min_distance: 新选角点与已有角点的最小间隔。
        quality_level: Shi-Tomasi 角点质量阈值。
        win_size, max_level: 金字塔 LK 的窗口边长和最高金字塔层级。
        fb_threshold: 前后向回投误差上限，单位为 SD 像素。
        redetect_interval: 周期性补点的 SD 帧间隔。
        max_trail_age: 失活轨迹继续显示的 SD 帧数，不影响跟踪。
    """

    def __init__(self, max_features=300, max_history=100, max_displacement=40,
                 min_distance=8, quality_level=0.01, win_size=21, max_level=3,
                 fb_threshold=1.5, redetect_interval=5, max_trail_age=100):
        if min(max_features, max_history, redetect_interval) < 1:
            raise ValueError("feature count, history length and detection interval must be positive")
        if max_displacement <= 0 or min_distance < 0 or fb_threshold <= 0:
            raise ValueError("distance limits must be positive")
        if not 0 < quality_level <= 1 or win_size < 3 or win_size % 2 == 0 or max_level < 0:
            raise ValueError("invalid OpenCV feature or LK parameters")
        if max_trail_age < 0:
            raise ValueError("trail age must be non-negative")
        self.max_features = max_features
        self.max_history = max_history
        self.max_displacement = max_displacement
        self.min_distance = min_distance
        self.quality_level = quality_level
        self.win_size = win_size
        self.max_level = max_level
        self.fb_threshold = fb_threshold
        self.redetect_interval = redetect_interval
        self.max_trail_age = max_trail_age
        self.lk_parameters = {
            "winSize": (win_size, win_size),
            "maxLevel": max_level,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        self.reset()

    def reset(self):
        """清空 LK 状态、活跃轨迹及仅用于可视化的历史轨迹。"""
        self.previous_image = None
        self.previous_points = np.empty((0, 1, 2), dtype=np.float32)
        self.sd_scale = None
        self.history = {}
        self.trail_history = {}
        self.trail_last_seen = {}
        self.active_track_ids = []
        self.next_track_id = 0
        self.tracking_count = 0

    def _advance_tracks(self, image):
        """从上一帧跟踪到当前帧，再反向跟踪以剔除不稳定的点。"""
        new_history = {}
        active_points = []
        active_ids = []
        if self.previous_image is not None and len(self.previous_points):
            # 前向、后向分别估计位移；反向结果应回到原始位置附近。
            next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_image, image, self.previous_points, None, **self.lk_parameters
            )
            if next_points is not None:
                back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    image, self.previous_image, next_points, None, **self.lk_parameters
                )
                if back_points is not None:
                    height, width = image.shape
                    for index, track_id in enumerate(self.active_track_ids):
                        # 同时检查 LK 状态、图像边界、位移和前后向误差。
                        if not forward_status[index, 0] or not backward_status[index, 0]:
                            continue
                        previous = self.previous_points[index, 0]
                        current = next_points[index, 0]
                        back = back_points[index, 0]
                        if not (np.isfinite(current).all() and np.isfinite(back).all()):
                            continue
                        if not (0 <= current[0] < width and 0 <= current[1] < height):
                            continue
                        if np.linalg.norm(current - previous) > self.max_displacement:
                            continue
                        if np.linalg.norm(back - previous) > self.fb_threshold:
                            continue
                        # 保存为公开接口约定的 (row, column)，而非 OpenCV 的 (x, y)。
                        point = (float(current[1]), float(current[0]))
                        new_history[track_id] = (self.history[track_id] + [point])[-self.max_history:]
                        active_points.append(current)
                        active_ids.append(track_id)
        self.history = new_history
        self.active_track_ids = active_ids
        self.previous_points = (np.asarray(active_points, dtype=np.float32).reshape(-1, 1, 2)
                                if active_points else np.empty((0, 1, 2), dtype=np.float32))

    def _detect_new_tracks(self, image):
        """在远离已有轨迹的位置补充 Shi-Tomasi 角点。"""
        remaining = self.max_features - len(self.active_track_ids)
        if remaining <= 0:
            return
        # 避开窗口无法完整落入的边界，并遮挡当前活跃点附近区域。
        mask = np.full(image.shape, 255, dtype=np.uint8)
        border = min(self.win_size // 2 + 1, min(image.shape) // 4)
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
        for point in self.previous_points[:, 0]:
            cv2.circle(mask, tuple(np.rint(point).astype(int)),
                       int(np.ceil(self.min_distance)), 0, -1)
        new_points = cv2.goodFeaturesToTrack(
            image, maxCorners=remaining, qualityLevel=self.quality_level,
            minDistance=self.min_distance, mask=mask, blockSize=7
        )
        if new_points is None:
            return
        for point in new_points[:, 0]:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.active_track_ids.append(track_id)
            self.history[track_id] = [(float(point[1]), float(point[0]))]
        self.previous_points = np.concatenate((self.previous_points, new_points), axis=0)

    def update(self, sd_img, viz=True):
        """输入单帧二维 SD 幅值，返回活跃轨迹及 BGR 可视化图。"""
        image, self.sd_scale = _normalize_sd_image(sd_img, self.sd_scale)
        # 输入尺寸变化时，旧坐标和金字塔都不能继续使用。
        if self.previous_image is not None and self.previous_image.shape != image.shape:
            self.reset()
            image, self.sd_scale = _normalize_sd_image(sd_img, self.sd_scale)
        self._advance_tracks(image)
        # 周期性补点；活跃点不足一半时提前补点，而不是重置已有轨迹。
        if (self.tracking_count % self.redetect_interval == 0
                or len(self.active_track_ids) < self.max_features // 2):
            self._detect_new_tracks(image)
        self.previous_image = image
        # 失活轨迹仅用于显示，保留 max_trail_age 帧后清除。
        for track_id, trajectory in self.history.items():
            if len(trajectory) > 1:
                self.trail_history[track_id] = trajectory.copy()
                self.trail_last_seen[track_id] = self.tracking_count
        expired = [track_id for track_id, last_seen in self.trail_last_seen.items()
                   if self.tracking_count - last_seen > self.max_trail_age]
        for track_id in expired:
            del self.trail_history[track_id]
            del self.trail_last_seen[track_id]
        self.tracking_count += 1

        visualization = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if viz:
            # 只绘制较长的近期轨迹，避免大量短暂角点遮挡 SD 图像。
            visible_trails = sorted(
                ((track_id, trajectory) for track_id, trajectory in self.trail_history.items()
                 if len(trajectory) >= 6),
                key=lambda item: (len(item[1]), self.trail_last_seen[item[0]]),
                reverse=True,
            )[:50]
            for track_id, trajectory in visible_trails:
                polyline = np.rint([(column, row) for row, column in trajectory])
                age = self.tracking_count - 1 - self.trail_last_seen[track_id]
                intensity = max(80, 255 - 2 * age)
                cv2.polylines(visualization, [polyline.astype(np.int32).reshape(-1, 1, 2)],
                              False, (0, intensity, 0), 1)
            for trajectory in self.history.values():
                row, column = trajectory[-1]
                cv2.circle(visualization, (int(round(column)), int(round(row))),
                           2, (0, 0, 255), -1)
        return self.history, visualization
