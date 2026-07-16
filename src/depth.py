import cv2
import numpy as np
import torch

import config

DEPTH_SKIP_FRAMES = config.DEPTH_SKIP_FRAMES


class DepthEstimator:
    """
    Monocular depth estimation using MiDaS (via torch.hub).

    Output convention (MiDaS inverse depth):
        Higher value  →  object is CLOSER
        Lower  value  →  object is FARTHER

    We invert and normalise to match our convention:
        Higher normalised value  →  FARTHER
        Lower  normalised value  →  CLOSER
    """

    def __init__(self):
        print("[DepthEstimator] Loading MiDaS — first run downloads weights (~100 MB)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            config.MIDAS_MODEL,
            trust_repo=True,
        )
        self.model = self.model.to(self.device).eval()

        transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )
        self.transform = transforms.small_transform

        print(f"[DepthEstimator] Running on {self.device}")

        self._cached_depth_map = None
        self._frame_count      = 0

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        self._frame_count += 1
        if (self._frame_count % DEPTH_SKIP_FRAMES != 0
                and self._cached_depth_map is not None):
            return self._cached_depth_map

        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb).to(self.device)

        with torch.no_grad():
            raw = self.model(input_tensor)
            raw = torch.nn.functional.interpolate(
                raw.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_np = raw.cpu().numpy().astype(np.float32)

        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 1e-6:
            depth_norm = 1.0 - (depth_np - d_min) / (d_max - d_min)
        else:
            depth_norm = np.ones_like(depth_np) * 0.5

        self._cached_depth_map = depth_norm
        return depth_norm

    def get_object_depth(self, depth_map: np.ndarray, bbox: tuple) -> float:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        bw, bh   = x2 - x1, y2 - y1
        mx, my   = int(bw * 0.2), int(bh * 0.2)
        rx1, rx2 = x1 + mx, x2 - mx
        ry1, ry2 = y1 + my, y2 - my

        region = depth_map[ry1:ry2, rx1:rx2]
        if region.size == 0:
            return float(np.median(depth_map[y1:y2, x1:x2]))
        return float(np.median(region))

    def get_depth_overlay(self, depth_map: np.ndarray) -> np.ndarray:
        vis = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)