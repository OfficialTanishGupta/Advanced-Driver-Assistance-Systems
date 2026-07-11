import numpy as np
from filterpy.kalman import KalmanFilter

TRAIL_LENGTH      = 20
PREDICTION_STEPS  = 30
DEPTH_HISTORY_LEN = 10   


def build_kalman_filter():
    """
    Constant-velocity Kalman filter.
    State:       [x, y, vx, vy]
    Measurement: [x, y]
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)
    kf.R  = np.eye(2) * 10.0
    kf.Q  = np.eye(4) * 0.1
    kf.Q[2, 2] = 1.0
    kf.Q[3, 3] = 1.0
    kf.P = np.eye(4) * 500.0
    return kf


class ObjectPredictor:
    """
    Per-object Kalman filter tracker + depth history manager.
    """

    def __init__(self):
        self.filters:       dict[int, KalmanFilter] = {}
        self.trails:        dict[int, list]          = {}
        self.predictions:   dict[int, list]          = {}
        self.depth_history: dict[int, list]          = {}   # NEW


    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _init_filter(self, track_id: int, cx: float, cy: float):
        kf = build_kalman_filter()
        kf.x = np.array([[cx], [cy], [0.0], [0.0]])
        self.filters[track_id]       = kf
        self.trails[track_id]        = []
        self.depth_history[track_id] = []


    def update(self, detections: list):
        """Update Kalman filters for all current detections."""
        active_ids = set()

        for det in detections:
            track_id = det["id"]
            cx, cy   = self._get_center(det["bbox"])
            active_ids.add(track_id)

            if track_id not in self.filters:
                self._init_filter(track_id, cx, cy)

            kf = self.filters[track_id]
            kf.predict()
            kf.update(np.array([[cx], [cy]]))

            smoothed_x = float(kf.x[0])
            smoothed_y = float(kf.x[1])
            self.trails[track_id].append((smoothed_x, smoothed_y))
            if len(self.trails[track_id]) > TRAIL_LENGTH:
                self.trails[track_id].pop(0)

            future     = []
            sim_state  = kf.x.copy()
            for _ in range(PREDICTION_STEPS):
                sim_state = kf.F @ sim_state
                future.append((float(sim_state[0]), float(sim_state[1])))
            self.predictions[track_id] = future

        for lost_id in set(self.filters.keys()) - active_ids:
            del self.filters[lost_id]
            del self.trails[lost_id]
            self.predictions.pop(lost_id, None)
            self.depth_history.pop(lost_id, None)

    def update_depth(self, track_id: int, depth_value: float):
        """
        Record a new depth reading for a tracked object.
        Called separately from update() after depth map is computed.
        """
        if track_id not in self.depth_history:
            self.depth_history[track_id] = []
        self.depth_history[track_id].append(depth_value)
        if len(self.depth_history[track_id]) > DEPTH_HISTORY_LEN:
            self.depth_history[track_id].pop(0)

    def get_depth_closing_rate(self, track_id: int) -> float:
        """
        Returns average depth units lost per frame (positive = approaching).
        DA V2: depth decreases as object comes closer, so
        closing_rate = prev_depth - current_depth per frame.
        Returns 0.0 if not enough history.
        """
        hist = self.depth_history.get(track_id, [])
        if len(hist) < 3:
            return 0.0
        rates = [hist[i - 1] - hist[i] for i in range(1, len(hist))]
        return float(np.mean(rates))

    def get_current_depth(self, track_id: int) -> float:
        hist = self.depth_history.get(track_id, [])
        return hist[-1] if hist else 0.5

    def get_trail(self, track_id: int):
        return self.trails.get(track_id, [])

    def get_prediction(self, track_id: int):
        return self.predictions.get(track_id, [])

    def get_velocity(self, track_id: int):
        if track_id not in self.filters:
            return (0.0, 0.0)
        kf = self.filters[track_id]
        return (float(kf.x[2]), float(kf.x[3]))