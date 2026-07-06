import cv2
import numpy as np

CLASS_COLORS = {
    "car":        (0, 255, 0),
    "motorcycle": (0, 200, 255),
    "bus":        (255, 0, 0),
    "truck":      (255, 100, 0),
    "person":     (0, 0, 255),
}

TRAIL_COLOR   = (180, 180, 180)
PREDICT_COLOR = (0, 255, 255)

RISK_COLORS = {
    "LOW":    (0, 255, 0),      # green
    "MEDIUM": (0, 165, 255),    # orange
    "HIGH":   (0, 0, 255),      # red
}


def draw_danger_zone(frame, zone_pts):
    """Draws a semi-transparent trapezoid representing our vehicle's path ahead."""
    overlay = frame.copy()
    pts = np.array(zone_pts, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (0, 255, 255))
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True,
                  color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return frame


def draw_detections(frame, detections, predictor=None, risk_scores=None, zone_pts=None):
    """
    Draws bounding boxes color-coded by risk, TTC label, trail, and predicted arrow.
    """
    if zone_pts is not None:
        frame = draw_danger_zone(frame, zone_pts)

    for det in detections:
        track_id        = det["id"]
        x1, y1, x2, y2 = map(int, det["bbox"])

        # Pick color: risk-based if available, else class-based
        risk_info = (risk_scores or {}).get(track_id, {})
        risk      = risk_info.get("risk", "LOW")
        ttc       = risk_info.get("ttc", float("inf"))
        color     = RISK_COLORS.get(risk, CLASS_COLORS.get(det["class_name"], (200, 200, 200)))

        # Bounding box — thicker for HIGH risk
        thickness = 3 if risk == "HIGH" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label: class + TTC
        ttc_str = f"{ttc:.1f}s" if ttc != float("inf") else "safe"
        label   = f"ID {track_id} {det['class_name']} | {risk} {ttc_str}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        if predictor is None:
            continue

        # Trail
        trail = predictor.get_trail(track_id)
        for i, (tx, ty) in enumerate(trail):
            alpha = int(80 + 175 * (i / max(len(trail), 1)))
            cv2.circle(frame, (int(tx), int(ty)), 2, (alpha, alpha, alpha), -1)

        # Predicted path arrow (color matches risk)
        future = predictor.get_prediction(track_id)
        if len(future) >= 2:
            pts_pred = [(int(fx), int(fy)) for fx, fy in future[::3]]
            for i in range(len(pts_pred) - 1):
                cv2.line(frame, pts_pred[i], pts_pred[i + 1], color, 1)
            if len(pts_pred) >= 2:
                cv2.arrowedLine(frame, pts_pred[-2], pts_pred[-1],
                                color, 2, tipLength=0.4)

    return frame


def draw_hud(frame, risk_scores: dict):
    """
    Draws a small heads-up panel in the top-left corner
    showing the highest current risk level.
    """
    if not risk_scores:
        return frame

    levels = [v["risk"] for v in risk_scores.values()]
    if "HIGH" in levels:
        overall, color = "HIGH RISK", (0, 0, 255)
    elif "MEDIUM" in levels:
        overall, color = "CAUTION", (0, 165, 255)
    else:
        overall, color = "CLEAR", (0, 255, 0)

    cv2.rectangle(frame, (10, 10), (200, 45), (0, 0, 0), -1)
    cv2.putText(frame, overall, (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return frame