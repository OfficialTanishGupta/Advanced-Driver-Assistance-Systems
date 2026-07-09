import argparse
import cv2
import winsound  # Windows built-in, no install needed

from detector import VehicleDetector
from tracker import draw_detections, draw_hud
from predictor import ObjectPredictor
from risk import RiskScorer

ALERT_COOLDOWN_FRAMES = 60


def parse_args():
    parser = argparse.ArgumentParser(description="Road Accident Predictor - Week 3")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video file path, '0' for webcam, or phone stream URL",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output to outputs/annotated.mp4",
    )
    parser.add_argument(
        "--hood-frac",
        type=float,
        default=None,
        help="Manually override hood line as a fraction of frame height (0.0-1.0). Skips auto-detection.",
    )
    parser.add_argument("--no-alert", action="store_true", help="Disable audio alerts")
    return parser.parse_args()


def main():
    args = parse_args()
    source = 0 if args.source == "0" else args.source

    detector = VehicleDetector(model_path="yolov8n.pt")
    predictor = ObjectPredictor()
    scorer = RiskScorer()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source {source}")
        return

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter("outputs/annotated.mp4", fourcc, fps, (width, height))

    alert_cooldown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]

        # Machine learning inference & telemetry calculations
        detections = detector.track(frame)
        predictor.update(detections)
        risk_scores, zone_pts = scorer.score(
            detections, predictor, fw, fh, frame=frame, hood_frac=args.hood_frac
        )

        # UI Overlay Compositing
        frame = draw_detections(
            frame,
            detections,
            predictor=predictor,
            risk_scores=risk_scores,
            zone_pts=zone_pts,
        )
        frame = draw_hud(frame, risk_scores)

        if not args.no_alert:
            high_risk = any(v["risk"] == "HIGH" for v in risk_scores.values())
            if high_risk and alert_cooldown == 0:
                winsound.Beep(1000, 200)  # 1000 Hz, 200ms
                alert_cooldown = ALERT_COOLDOWN_FRAMES
            if alert_cooldown > 0:
                alert_cooldown -= 1

        cv2.imshow("Road Accident Predictor", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
