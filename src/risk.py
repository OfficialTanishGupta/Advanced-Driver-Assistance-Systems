import cv2
import numpy as np

import config

DANGER_ZONE_FRAC = config.DANGER_ZONE_FRAC
TTC_HIGH         = config.TTC_HIGH
TTC_MEDIUM       = config.TTC_MEDIUM
FPS_ESTIMATE     = config.FPS_ESTIMATE

RISK_HIGH   = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_LOW    = "LOW"

FPS_ESTIMATE = 30.0


def get_danger_zone_pts(frame_w: int, frame_h: int):
    def scale(fx, fy):
        return (int(fx * frame_w), int(fy * frame_h))
    return [
        scale(*DANGER_ZONE_FRAC["tl"]),
        scale(*DANGER_ZONE_FRAC["tr"]),
        scale(*DANGER_ZONE_FRAC["br"]),
        scale(*DANGER_ZONE_FRAC["bl"]),
    ]


def point_in_trapezoid(px: float, py: float, pts: list) -> bool:
    poly   = np.array(pts, dtype=np.float32)
    result = cv2.pointPolygonTest(poly, (float(px), float(py)), False)
    return result >= 0


def compute_ttc(depth_value: float, closing_rate: float) -> float:
    """
    Depth-based TTC (replaces old bbox-height heuristic).

    depth_value   : current normalised depth (0–1, higher = farther)
    closing_rate  : depth units lost per frame (positive = approaching)

    TTC = depth_value / (closing_rate * FPS)
    Returns inf if object is stationary or moving away.
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
    Evaluates risk per detection using depth-based TTC.
    Requires predictor to have up-to-date depth history.
    """

    def score(self, detections: list, predictor, frame_w: int, frame_h: int) -> dict:
        zone_pts = get_danger_zone_pts(frame_w, frame_h)
        results  = {}

        for det in detections:
            track_id = det["id"]
            bbox     = det["bbox"]
            future   = predictor.get_prediction(track_id)

            # Zone check: current centre + first 15 predicted positions
            x1, y1, x2, y2 = bbox
            cx, cy  = (x1 + x2) / 2, (y1 + y2) / 2
            in_zone = point_in_trapezoid(cx, cy, zone_pts)
            if not in_zone:
                for fx, fy in future[:15]:
                    if point_in_trapezoid(fx, fy, zone_pts):
                        in_zone = True
                        break

            # Depth-based TTC  ← main upgrade from Week 3
            depth_value   = predictor.get_current_depth(track_id)
            closing_rate  = predictor.get_depth_closing_rate(track_id)
            ttc           = compute_ttc(depth_value, closing_rate)
            risk          = classify_risk(ttc, in_zone)

            results[track_id] = {
                "risk":         risk,
                "ttc":          ttc,
                "in_zone":      in_zone,
                "depth":        round(depth_value, 3),
                "closing_rate": round(closing_rate, 4),
            }

        return results