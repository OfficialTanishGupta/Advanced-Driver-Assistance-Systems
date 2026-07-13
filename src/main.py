import argparse
import cv2
import winsound

from detector import VehicleDetector
from tracker import draw_detections, draw_hud
from predictor import ObjectPredictor
from risk import RiskScorer, get_danger_zone_pts
from depth import DepthEstimator

ALERT_COOLDOWN_FRAMES = 60


def parse_args():
    parser = argparse.ArgumentParser(
        description="Road Accident Predictor — Depth Edition"
    )
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-alert", action="store_true")
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Show colourised depth map in a second window",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = 0 if args.source == "0" else args.source

    detector = VehicleDetector(model_path="yolov8n.pt")
    predictor = ObjectPredictor()
    scorer = RiskScorer()
    depth_est = DepthEstimator()  # loads DA V2 Small on startup

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source {source}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        writer = cv2.VideoWriter(
            "outputs/annotated.mp4", fourcc, fps, (frame_w, frame_h)
        )

    alert_cooldown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]

        # 1. Detect + track
        detections = detector.track(frame)

        # 2. Update Kalman filters
        predictor.update(detections)

        # 3. Estimate depth map
        depth_map = depth_est.estimate(frame)

        # 4. Feed per-object depth into predictor history
        for det in detections:
            depth_val = depth_est.get_object_depth(depth_map, det["bbox"])
            predictor.update_depth(det["id"], depth_val)

        # 5. Score risk using depth-based TTC
        risk_scores = scorer.score(detections, predictor, fw, fh)
        zone_pts = get_danger_zone_pts(fw, fh)

        # 6. Draw
        frame = draw_detections(
            frame,
            detections,
            predictor=predictor,
            risk_scores=risk_scores,
            zone_pts=zone_pts,
        )
        frame = draw_hud(frame, risk_scores)

        # 7. Audio alert
        if not args.no_alert:
            high_risk = any(v["risk"] == "HIGH" for v in risk_scores.values())
            if high_risk and alert_cooldown == 0:
                winsound.Beep(1000, 200)
                alert_cooldown = ALERT_COOLDOWN_FRAMES
            if alert_cooldown > 0:
                alert_cooldown -= 1

        cv2.imshow("Road Accident Predictor", frame)

        # Optional: show raw depth map for debugging
        if args.show_depth:
            depth_vis = depth_est.get_depth_overlay(depth_map)
            cv2.imshow("Depth Map (blue=far, red=close)", depth_vis)

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
