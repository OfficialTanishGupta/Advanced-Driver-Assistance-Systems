import cv2
import numpy as np
import config
from typing import Optional  # Compatible with all Python 3.x versions

TTC_HIGH = config.TTC_HIGH
TTC_MEDIUM = config.TTC_MEDIUM
FPS_ESTIMATE = config.FPS_ESTIMATE

RISK_HIGH = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_LOW = "LOW"


def get_danger_zone_pts(frame_w: int, frame_h: int):
    """Fixed fallback trapezoid from config fractions."""
    frac = config.DANGER_ZONE_FRAC

    def scale(fx, fy):
        return (int(fx * frame_w), int(fy * frame_h))

    return [
        scale(*frac["tl"]),
        scale(*frac["tr"]),
        scale(*frac["br"]),
        scale(*frac["bl"]),
    ]


def point_in_trapezoid(px: float, py: float, pts: list) -> bool:
    poly = np.array(pts, dtype=np.float32)
    result = cv2.pointPolygonTest(poly, (float(px), float(py)), False)
    return result >= 0


def compute_ttc(depth_value: float, closing_rate: float) -> float:
    """
    Depth-based TTC in seconds.
    depth_value  : normalised depth 0-1 (higher = farther)
    closing_rate : depth units lost per frame (positive = approaching)
    """
    if closing_rate < 0.001:
        return float("inf")
    ttc_frames = depth_value / closing_rate
    return round(ttc_frames / FPS_ESTIMATE, 2)


def classify_risk(ttc: float, in_zone: bool) -> str:
    if not in_zone:
        return RISK_LOW
    if ttc <= TTC_HIGH:
        return RISK_HIGH
    if ttc <= TTC_MEDIUM:
        return RISK_MEDIUM
    return RISK_LOW


class RiskScorer:
    """
    Evaluates risk per detection.
    Accepts zone_pts externally so LaneDetector can supply dynamic points.
    """

    # Changed from 'list | None' to 'Optional[list]' for Python < 3.10 support
    def score(
        self,
        detections: list,
        predictor,
        frame_w: int,
        frame_h: int,
        zone_pts: Optional[list] = None,
    ) -> dict:
        # Determine the final active points list
        active_zone_pts = (
            zone_pts if zone_pts is not None else get_danger_zone_pts(frame_w, frame_h)
        )

        results = {}
        for det in detections:
            track_id = det["id"]
            bbox = det["bbox"]

            future = predictor.get_prediction(track_id)
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            in_zone = point_in_trapezoid(cx, cy, active_zone_pts)

            future_points = future if future is not None else []

            if not in_zone:
                for fx, fy in future_points[:15]:
                    if point_in_trapezoid(fx, fy, active_zone_pts):
                        in_zone = True
                        break

            depth_value = predictor.get_current_depth(track_id)
            closing_rate = predictor.get_depth_closing_rate(track_id)
            ttc = compute_ttc(depth_value, closing_rate)
            risk = classify_risk(ttc, in_zone)

            results[track_id] = {
                "risk": risk,
                "ttc": ttc,
                "in_zone": in_zone,
                "depth": round(depth_value, 3),
                "closing_rate": round(closing_rate, 4),
            }
        return results
