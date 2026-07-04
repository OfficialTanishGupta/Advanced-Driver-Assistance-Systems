import numpy as np
from collections import defaultdict
from filterpy.kalman import KalmanFilter


class ObjectPredictor:
    def __init__(self, max_trail_len=30, pred_steps=30):
        self.max_trail_len = max_trail_len
        self.pred_steps = pred_steps
        self.filters = {}
        self.trails = defaultdict(list)

    def _init_kf(self, x, y):
        """Initializes a constant velocity Kalman Filter for a 2D coordinate."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        # State vector [x, y, vx, vy]
        kf.x = np.array([[x], [y], [0.0], [0.0]])

        # State transition matrix
        kf.F = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        kf.P *= 1000.0
        kf.R *= 10.0
        kf.Q *= 0.01
        return kf

    def update(self, detections):
        """Updates tracks and returns smoothed trajectories."""
        active_ids = set()

        for det in detections:
            track_id = det["id"]
            active_ids.add(track_id)

            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if track_id not in self.filters:
                self.filters[track_id] = self._init_kf(cx, cy)

            kf = self.filters[track_id]
            kf.predict()
            kf.update(np.array([[cx], [cy]]))

            smoothed_x = float(kf.x[0, 0])
            smoothed_y = float(kf.x[1, 0])

            self.trails[track_id].append((smoothed_x, smoothed_y))
            if len(self.trails[track_id]) > self.max_trail_len:
                self.trails[track_id].pop(0)

        # Clean up dead tracks
        dead_ids = set(self.filters.keys()) - active_ids
        for dead_id in dead_ids:
            del self.filters[dead_id]
            if dead_id in self.trails:
                del self.trails[dead_id]

    def get_trail(self, track_id):
        """Returns the list of historical coordinates for a given track."""
        return self.trails.get(track_id, [])

    def get_prediction(self, track_id):
        """Projects the trajectory into the future based on current velocity."""
        if track_id not in self.filters:
            return []

        kf = self.filters[track_id]
        px = float(kf.x[0, 0])
        py = float(kf.x[1, 0])
        vx = float(kf.x[2, 0])
        vy = float(kf.x[3, 0])

        predictions = []
        for step in range(1, self.pred_steps + 1):
            pred_x = px + vx * step
            pred_y = py + vy * step
            predictions.append((pred_x, pred_y))

        return predictions
