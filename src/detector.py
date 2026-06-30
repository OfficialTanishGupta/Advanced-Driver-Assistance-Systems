from ultralytics import YOLO

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    0: "person",
}


class VehicleDetector:
    """Wraps a YOLOv8 model and filters detections to vehicle/pedestrian classes."""

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def track(self, frame):
        """
        Runs detection + tracking on a single frame.
        Returns a list of dicts: {id, class_name, bbox, confidence}
        bbox is (x1, y1, x2, y2) in pixel coordinates.
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_CLASSES.keys()),
            conf=self.conf_threshold,
            verbose=False,
        )

        detections = []
        if results[0].boxes.id is None:
            return detections

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
            detections.append({
                "id": int(track_id),
                "class_name": VEHICLE_CLASSES.get(cls_id, "unknown"),
                "bbox": tuple(box),
                "confidence": float(conf),
            })

        return detections