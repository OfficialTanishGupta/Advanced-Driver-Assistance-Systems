YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.4

MIDAS_MODEL = "MiDaS_small"
DEPTH_SKIP_FRAMES = 2


TRAIL_LENGTH = 20  # past positions to remember per object
PREDICTION_STEPS = 30  # frames ahead to project (~1 sec at 30fps)
DEPTH_HISTORY_LEN = 10


DANGER_ZONE_FRAC = {
    "tl": (0.35, 0.40),  # top-left
    "tr": (0.65, 0.40),  # top-right
    "br": (0.80, 0.90),  # bottom-right
    "bl": (0.20, 0.90),  # bottom-left
}


TTC_HIGH = 2.0  # seconds — below this → HIGH risk
TTC_MEDIUM = 4.0  # seconds — below this → MEDIUM risk
FPS_ESTIMATE = 30.0

ALERT_COOLDOWN_FRAMES = 60
ALERT_FREQ_HZ = 1000  # beep frequency
ALERT_DURATION_MS = 500  # beep duration

LOG_ENABLED = True
LOG_PATH = "outputs/risk_log.csv"
LOG_MIN_RISK = "HIGH"
