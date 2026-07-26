import csv
import os
from datetime import datetime

import config

FIELDNAMES = [
    "timestamp",
    "frame",
    "track_id",
    "class_name",
    "risk",
    "ttc_seconds",
    "depth",
    "closing_rate",
    "in_zone",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
]

# Risk level ordering for filtering
RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


class RiskLogger:
    """
    Logs risk events to a CSV file after each run.
    Only records events at or above config.LOG_MIN_RISK level.
    """

    def __init__(self, log_path: str = config.LOG_PATH):
        self.log_path = log_path
        self.enabled = config.LOG_ENABLED
        self.min_risk = config.LOG_MIN_RISK
        self._buffer = []  # holds rows until flush
        self._frame_num = 0

        if self.enabled:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            # Write header (overwrite previous log each run)
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
            print(f"[RiskLogger] Logging to {log_path}")

    def tick(self):
        """Call once per frame to advance the frame counter."""
        self._frame_num += 1

    def log(self, detections: list, risk_scores: dict):
        """
        Called once per frame with current detections + risk scores.
        Buffers qualifying events; flushes every 30 frames for efficiency.
        """
        if not self.enabled:
            return

        min_order = RISK_ORDER.get(self.min_risk, 2)

        for det in detections:
            track_id = det["id"]
            risk_info = risk_scores.get(track_id, {})
            risk = risk_info.get("risk", "LOW")

            if RISK_ORDER.get(risk, 0) < min_order:
                continue

            x1, y1, x2, y2 = det["bbox"]
            self._buffer.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "frame": self._frame_num,
                    "track_id": track_id,
                    "class_name": det["class_name"],
                    "risk": risk,
                    "ttc_seconds": risk_info.get("ttc", "inf"),
                    "depth": risk_info.get("depth", ""),
                    "closing_rate": risk_info.get("closing_rate", ""),
                    "in_zone": risk_info.get("in_zone", ""),
                    "bbox_x1": round(x1, 1),
                    "bbox_y1": round(y1, 1),
                    "bbox_x2": round(x2, 1),
                    "bbox_y2": round(y2, 1),
                }
            )

        if self._frame_num % 30 == 0:
            self._flush()

    def _flush(self):
        """Write buffered rows to CSV."""
        if not self._buffer:
            return
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerows(self._buffer)
        self._buffer.clear()

    def close(self):
        """Flush remaining buffer on shutdown."""
        self._flush()
        if self.enabled:
            print(f"[RiskLogger] Log saved → {self.log_path}")
    