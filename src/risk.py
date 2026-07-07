import cv2
import numpy as np

DANGER_ZONE_FRAC = {
    "tl": (0.35, 0.32),  # Left point near the mountain road horizon
    "tr": (0.52, 0.32),  # Right point near the oncoming lane horizon
}

# TTC thresholds in seconds
TTC_HIGH = 2.0
TTC_MEDIUM = 4.0

RISK_HIGH = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_LOW = "LOW"


def detect_hood_line(frame: np.ndarray, default_y_frac: float = 0.52) -> int:
    """
    Dynamically locates the topmost boundary of the car hood/dashboard assembly.
    Focuses on a central horizontal slot to bypass steering wheel curves.
    """
    if frame is None:
        return int(default_y_frac * 480)

    fh, fw = frame.shape[:2]

    x_start = int(fw * 0.25)
    x_end = int(fw * 0.60)

    y_start = int(fh * 0.40)
    roi = frame[y_start:fh, x_start:x_end]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 80, 200)

    horizontal_profile = np.sum(edges > 0, axis=1)

    min_edge_pixels = int((x_end - x_start) * 0.15)
    detected_y_offset = int((fh - y_start) * 0.35)  # Safe initial default offset

    for y, edge_count in enumerate(horizontal_profile):
        if edge_count > min_edge_pixels:
            detected_y_offset = y
            break

    hood_y = y_start + detected_y_offset

=    min_allowed_y = int(fh * 0.42)
    max_allowed_y = int(fh * 0.58)

    return np.clip(hood_y - 8, min_allowed_y, max_allowed_y)


def get_danger_zone_pts(frame_w: int, frame_h: int, frame: np.ndarray = None):
    """
    Calculates danger zone coordinates by locking the horizon to fixed scales
    and dynamically snapping the base points right above the vehicle's hood line.
    """
    # 1. Compute the structural dashboard mask line elevation
    hood_y = detect_hood_line(frame) if frame is not None else int(0.50 * frame_h)

    # 2. Extract top tracking points using standard coordinate percentages
    top_left_x = int(DANGER_ZONE_FRAC["tl"][0] * frame_w)
    top_left_y = int(DANGER_ZONE_FRAC["tl"][1] * frame_h)

    top_right_x = int(DANGER_ZONE_FRAC["tr"][0] * frame_w)
    top_right_y = int(DANGER_ZONE_FRAC["tr"][1] * frame_h)

    # 3. Project base width geometry out flush with the calculated edge plane
    bottom_right_x = int(0.58 * frame_w)
    bottom_left_x = int(0.24 * frame_w)

    return [
        (top_left_x, top_left_y),
        (top_right_x, top_right_y),
        (bottom_right_x, int(hood_y)),
        (bottom_left_x, int(hood_y)),
    ]


def point_in_trapezoid(px: float, py: float, pts: list) -> bool:
    """Returns True if point (px, py) is inside the trapezoid defined by pts."""
    poly = np.array(pts, dtype=np.float32)
    result = cv2.pointPolygonTest(poly, (float(px), float(py)), False)
    return result >= 0


def compute_ttc(bbox, velocity, frame_h: int) -> float:
    """Heuristic Time-To-Collision calculation in seconds."""
    x1, y1, x2, y2 = bbox
    vx, vy = velocity

    if vy <= 0.5:
        return float("inf")

    fps_estimate = 30.0
    vy_per_sec = vy * fps_estimate
    remaining = max(frame_h - y2, 1)

    return round(remaining / vy_per_sec, 2)


def classify_risk(ttc: float, in_zone: bool) -> str:
    """Combines zone presence and TTC into a single risk level."""
    if not in_zone:
        return RISK_LOW
    if ttc <= TTC_HIGH:
        return RISK_HIGH
    if ttc <= TTC_MEDIUM:
        return RISK_MEDIUM
    return RISK_LOW


class RiskScorer:
    """Evaluates risk profiles for tracked objects frame-by-frame."""

    def score(
        self,
        detections: list,
        predictor,
        frame_w: int,
        frame_h: int,
        frame: np.ndarray = None,
    ) -> dict:
        zone_pts = get_danger_zone_pts(frame_w, frame_h, frame)
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
