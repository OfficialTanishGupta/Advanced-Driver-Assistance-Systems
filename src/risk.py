import cv2
import numpy as np
from collections import deque
from typing import Optional

DANGER_ZONE_FRAC = {
    "tl": (0.35, 0.32),  # fallback only — used when lane detection fails
    "tr": (0.52, 0.32),
}

# TTC thresholds in seconds
TTC_HIGH = 2.0
TTC_MEDIUM = 4.0

RISK_HIGH = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_LOW = "LOW"


def detect_hood_line(
    frame: Optional[np.ndarray] = None, default_y_frac: float = 0.78
) -> int:
    """
    Dynamically locates the top boundary of the dashboard/hood by scanning
    upward from the bottom of the frame for a stable low-variance region
    (the hood/dashboard is typically a large, uniformly colored surface,
    unlike the textured road/scenery above it).
    """
    if frame is None:
        return int(default_y_frac * 480)

    fh, fw = frame.shape[:2]

    x_start = int(fw * 0.20)
    x_end = int(fw * 0.65)
    search_top = int(fh * 0.45)

    roi = frame[search_top:fh, x_start:x_end]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    row_std = gray.std(axis=1)
    roi_h = roi.shape[0]

    win = max(int(roi_h * 0.03), 3)
    kernel = np.ones(win) / win
    smoothed = np.convolve(row_std, kernel, mode="same")

    low_thresh = np.percentile(smoothed, 25) * 1.6

    hood_offset = roi_h - 1
    consecutive_high = 0
    for y in range(roi_h - 1, -1, -1):
        if smoothed[y] > low_thresh:
            consecutive_high += 1
            if consecutive_high > win * 2:
                hood_offset = y + consecutive_high
                break
        else:
            consecutive_high = 0
            hood_offset = y

    hood_y = search_top + hood_offset

    min_allowed_y = int(fh * 0.45)
    max_allowed_y = int(fh * 0.93)  # was 0.58 — this was the bug
    return int(np.clip(hood_y, min_allowed_y, max_allowed_y))


class LaneZoneTracker:
    """
    Detects the ego-lane boundaries per frame using Canny edges + Hough line
    detection, then builds the danger-zone trapezoid from those boundaries so
    it follows the actual road/lane as it curves, instead of a fixed
    screen-space region.

    Falls back to the fixed DANGER_ZONE_FRAC geometry when lane lines can't
    be reliably found (e.g. faded markings, glare, sharp turns), and smooths
    results across frames with an exponential moving average to avoid jitter.
    """

    def __init__(self, smoothing: float = 0.75, history_len: int = 5):
        self.smoothing = smoothing
        self.left_history = deque(maxlen=history_len)
        self.right_history = deque(maxlen=history_len)
        self.last_left = None  # (slope, intercept) fit as x = m*y + b
        self.last_right = None

    @staticmethod
    def _fit_line_xy(segments):
        """Fit x = m*y + b across a list of (x1, y1, x2, y2) segments."""
        xs, ys = [], []
        for x1, y1, x2, y2 in segments:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        if len(xs) < 2:
            return None
        m, b = np.polyfit(ys, xs, 1)
        return float(m), float(b)

    def _detect_raw_segments(self, frame):
        fh, fw = frame.shape[:2]

        roi_top = int(fh * 0.45)
        roi_bottom = int(fh * 0.90)
        roi = frame[roi_top:roi_bottom, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=int(fh * 0.06),
            maxLineGap=int(fh * 0.05),
        )

        left_segs, right_segs = [], []
        if lines is None:
            return left_segs, right_segs

        cx = fw / 2.0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:  # discard near-horizontal noise (not lane lines)
                continue

            y1f, y2f = y1 + roi_top, y2 + roi_top
            mid_x = (x1 + x2) / 2.0
            seg = (x1, y1f, x2, y2f)

            if slope < 0 and mid_x < cx:
                left_segs.append(seg)
            elif slope > 0 and mid_x >= cx:
                right_segs.append(seg)

        return left_segs, right_segs

    def _update_side(self, segments, last_fit, history):
        fit = self._fit_line_xy(segments)
        if fit is None:
            return last_fit  # no detection this frame, keep previous fit

        history.append(fit)
        m_vals = [f[0] for f in history]
        b_vals = [f[1] for f in history]
        m_avg = sum(m_vals) / len(m_vals)
        b_avg = sum(b_vals) / len(b_vals)

        if last_fit is None:
            return (m_avg, b_avg)

        m = self.smoothing * last_fit[0] + (1 - self.smoothing) * m_avg
        b = self.smoothing * last_fit[1] + (1 - self.smoothing) * b_avg
        return (m, b)

    def get_zone_pts(self, frame_w, frame_h, frame, hood_y):
        y_top = int(0.32 * frame_h)
        y_bottom = int(hood_y)

        if frame is not None:
            left_segs, right_segs = self._detect_raw_segments(frame)
            self.last_left = self._update_side(
                left_segs, self.last_left, self.left_history
            )
            self.last_right = self._update_side(
                right_segs, self.last_right, self.right_history
            )

        if self.last_left is not None and self.last_right is not None:
            lm, lb = self.last_left
            rm, rb = self.last_right

            top_left_x = int(lm * y_top + lb)
            bottom_left_x = int(lm * y_bottom + lb)
            top_right_x = int(rm * y_top + rb)
            bottom_right_x = int(rm * y_bottom + rb)

            valid = (
                0 <= top_left_x < top_right_x <= frame_w
                and 0 <= bottom_left_x < bottom_right_x <= frame_w
                and bottom_left_x < top_left_x + int(0.4 * frame_w)
            )
            if valid:
                return [
                    (top_left_x, y_top),
                    (top_right_x, y_top),
                    (bottom_right_x, y_bottom),
                    (bottom_left_x, y_bottom),
                ]

        # Fallback: original fixed-fraction geometry
        top_left_x = int(DANGER_ZONE_FRAC["tl"][0] * frame_w)
        top_right_x = int(DANGER_ZONE_FRAC["tr"][0] * frame_w)
        bottom_right_x = int(0.58 * frame_w)
        bottom_left_x = int(0.24 * frame_w)
        return [
            (top_left_x, y_top),
            (top_right_x, y_top),
            (bottom_right_x, y_bottom),
            (bottom_left_x, y_bottom),
        ]


_lane_tracker = LaneZoneTracker()


def get_danger_zone_pts(
    frame_w: int,
    frame_h: int,
    frame: Optional[np.ndarray] = None,
    hood_frac: Optional[float] = None,
) -> list:
    if hood_frac is not None:
        hood_y = int(hood_frac * frame_h)
    else:
        hood_y = detect_hood_line(frame) if frame is not None else int(0.50 * frame_h)
    return _lane_tracker.get_zone_pts(frame_w, frame_h, frame, hood_y)


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
        frame: Optional[np.ndarray] = None,
        hood_frac: Optional[float] = None,
    ) -> tuple:
        zone_pts = get_danger_zone_pts(frame_w, frame_h, frame, hood_frac=hood_frac)
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

        return results, zone_pts
