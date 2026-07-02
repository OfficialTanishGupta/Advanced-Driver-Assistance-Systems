import numpy as np
from filterpy.kalman import KalmanFilter

TRAIL_LENGTH = 20

PREDICTION_STEPS = 30  # at 30fps this is ~1 second ahead


def build_kalman_filter():
    """
    Builds a constant-velocity Kalman filter.
    State vector:      [x, y, vx, vy]
    Measurement vector: [x, y]   (centre of bounding box)
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State transition matrix: assumes constant velocity
    # x_new  = x + vx
    # y_new  = y + vy
    # vx_new = vx
    # vy_new = vy
    kf.F = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )

    # Measurement matrix: we only observe [x, y], not velocity
    kf.H = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=float,
    )

    # Measurement noise covariance — how much we trust the detector
    kf.R = np.eye(2) * 10.0

    # Process noise covariance — how much we expect the object to change velocity
    kf.Q = np.eye(4) * 0.1
    kf.Q[2, 2] = 1.0  # vx can change more
    kf.Q[3, 3] = 1.0  # vy can change more

    # Initial covariance — high uncertainty at start
    kf.P = np.eye(4) * 500.0

    return kf


class ObjectPredictor:
    """
    Manages one Kalman filter per tracked object ID.
    Maintains position history and produces future trajectory predictions.
    """

    def __init__(self):
        self.filters: dict[int, KalmanFilter] = {}
        self.trails: dict[int, list] = {}  # past positions
        self.predictions: dict[int, list] = {}  # future predicted positions

    def _get_center(self, bbox):
        """Returns (cx, cy) from (x1, y1, x2, y2) bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _init_filter(self, track_id: int, cx: float, cy: float):
        """Creates and initialises a new Kalman filter for a new track ID."""
        kf = build_kalman_filter()
        kf.x = np.array([[cx], [cy], [0.0], [0.0]])  # start with zero velocity
        self.filters[track_id] = kf
        self.trails[track_id] = []

    def update(self, detections: list):
        """
        Call once per frame with the list of current detections.
        Updates each object's Kalman filter and records trail + predictions.
        """
        active_ids = set()

        for det in detections:
            track_id = det["id"]
            cx, cy = self._get_center(det["bbox"])
            active_ids.add(track_id)

            if track_id not in self.filters:
                self._init_filter(track_id, cx, cy)

            kf = self.filters[track_id]
            kf.predict()
            kf.update(np.array([[cx], [cy]]))

            # Record smoothed position in trail
            smoothed_x = float(kf.x[0])
            smoothed_y = float(kf.x[1])
            self.trails[track_id].append((smoothed_x, smoothed_y))

            # Keep trail at fixed length
            if len(self.trails[track_id]) > TRAIL_LENGTH:
                self.trails[track_id].pop(0)

            # Project future positions forward by cloning current state
            future = []
            sim_state = kf.x.copy()
            for _ in range(PREDICTION_STEPS):
                sim_state = kf.F @ sim_state
                future.append((float(sim_state[0]), float(sim_state[1])))
            self.predictions[track_id] = future

        # Clean up filters for IDs that have disappeared
        lost_ids = set(self.filters.keys()) - active_ids
        for lost_id in lost_ids:
            del self.filters[lost_id]
            del self.trails[lost_id]
            self.predictions.pop(lost_id, None)

    def get_trail(self, track_id: int):
        return self.trails.get(track_id, [])

    def get_prediction(self, track_id: int):
        return self.predictions.get(track_id, [])

    def get_velocity(self, track_id: int):
        """Returns (vx, vy) in pixels/frame for a given track ID."""
        if track_id not in self.filters:
            return (0.0, 0.0)
        kf = self.filters[track_id]
        return (float(kf.x[2]), float(kf.x[3]))
