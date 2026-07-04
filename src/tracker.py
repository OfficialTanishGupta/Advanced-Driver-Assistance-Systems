import cv2

CLASS_COLORS = {
    "car": (0, 255, 0),
    "motorcycle": (0, 200, 255),
    "bus": (255, 0, 0),
    "truck": (255, 100, 0),
    "person": (0, 0, 255),
}

TRAIL_COLOR = (180, 180, 180)
PREDICT_COLOR = (0, 255, 255)


def draw_detections(frame, detections, predictor=None):
    for det in detections:
        track_id = det["id"]
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = CLASS_COLORS.get(det["class_name"], (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {track_id} {det['class_name']} {det['confidence']:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        if predictor is None:
            continue

        trail = predictor.get_trail(track_id)
        for i, (tx, ty) in enumerate(trail):
            alpha = int(80 + 175 * (i / max(len(trail), 1)))
            cv2.circle(frame, (int(tx), int(ty)), 2, (alpha, alpha, alpha), -1)

        future = predictor.get_prediction(track_id)
        if len(future) >= 2:
            pts = [(int(fx), int(fy)) for fx, fy in future[::3]]
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], PREDICT_COLOR, 1)
            if len(pts) >= 2:
                cv2.arrowedLine(
                    frame, pts[-2], pts[-1], PREDICT_COLOR, 2, tipLength=0.4
                )

    return frame
