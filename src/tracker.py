import cv2

# Distinct colors per risk level later; for now just one color per class
CLASS_COLORS = {
    "car": (0, 255, 0),
    "motorcycle": (0, 200, 255),
    "bus": (255, 0, 0),
    "truck": (255, 100, 0),
    "person": (0, 0, 255),
}


def draw_detections(frame, detections):
    """Draws bounding boxes, track IDs, and class labels on the frame."""
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = CLASS_COLORS.get(det["class_name"], (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {det['id']} {det['class_name']} {det['confidence']:.2f}"
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

    return frame
