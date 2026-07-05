import numpy as np

DANGER_ZONE_FRAC = {
    "tl": (0.35, 0.40),
    "tr": (0.65, 0.40),
    "br": (0.80, 0.90),
    "bl": (0.20, 0.90),
}

# TTC thresholds in seconds
TTC_HIGH = 2.0  # below this → RED
TTC_MEDIUM = 4.0  # below this → YELLOW
# above      → GREEN

RISK_HIGH = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_LOW = "LOW"


def get_danger_zone_pts(frame_w: int, frame_h: int):
    """Returns the 4 danger zone corners as integer pixel coordinates."""

    def scale(frac_x, frac_y):
        return (int(frac_x * frame_w), int(frac_y * frame_h))

    return [
        scale(*DANGER_ZONE_FRAC["tl"]),
        scale(*DANGER_ZONE_FRAC["tr"]),
        scale(*DANGER_ZONE_FRAC["br"]),
        scale(*DANGER_ZONE_FRAC["bl"]),
    ]


def point_in_trapezoid(px: float, py: float, pts: list) -> bool:
    """
    Returns True if point (px, py) is inside the trapezoid defined by pts.
    Uses OpenCV-style polygon point test via cross-product winding.
    """
    import cv2
    import numpy as np

    poly = np.array(pts, dtype=np.float32)
    result = cv2.pointPolygonTest(poly, (float(px), float(py)), False)
    return result >= 0


def compute_ttc(bbox, velocity, frame_h: int) -> float:
    """
    Heuristic Time-To-Collision in seconds.

    Logic:
    - bbox height in pixels is a proxy for distance (bigger = closer).
    - vy (pixels/frame downward) means object is approaching our camera.
    - TTC ≈ (remaining_pixels_to_close) / vy_in_pixels_per_second

    Returns float('inf') if the object is moving away or stationary.
    """
    x1, y1, x2, y2 = bbox
    bbox_h = y2 - y1
    vx, vy = velocity

    if vy <= 0.5:
        return float("inf")

    fps_estimate = 30.0
    vy_per_sec = vy * fps_estimate

    remaining = max(frame_h - y2, 1)

    ttc = remaining / vy_per_sec
    return round(ttc, 2)


def classify_risk(ttc: float, in_zone: bool) -> str:
    """
    Combines zone presence and TTC into a single risk level.
    If not in danger zone at all → LOW regardless of TTC.
    """
    if not in_zone:
        return RISK_LOW
    if ttc <= TTC_HIGH:
        return RISK_HIGH
    if ttc <= TTC_MEDIUM:
        return RISK_MEDIUM
    return RISK_LOW


class RiskScorer:
    """
    Evaluates risk for every tracked detection each frame.
    Returns a dict: { track_id: { "risk": str, "ttc": float, "in_zone": bool } }
    """

    def score(self, detections: list, predictor, frame_w: int, frame_h: int) -> dict:
        zone_pts = get_danger_zone_pts(frame_w, frame_h)
        results = {}

        for det in detections:
            track_id = det["id"]
            bbox = det["bbox"]
            velocity = predictor.get_velocity(track_id)
            future = predictor.get_prediction(track_id)

            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            in_zone = point_in_trapezoid(cx, cy, zone_pts)
            if not in_zone:
                for fx, fy in future[:15]:
                    if point_in_trapezoid(fx, fy, zone_pts):
                        in_zone = True
                        break

            ttc = compute_ttc(bbox, velocity, frame_h)
            risk = classify_risk(ttc, in_zone)

            results[track_id] = {
                "risk": risk,
                "ttc": ttc,
                "in_zone": in_zone,
            }

        return results
