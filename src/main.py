import argparse
import cv2
import winsound
import threading

import config
from detector import VehicleDetector
from tracker import draw_detections, draw_hud
from predictor import ObjectPredictor
from risk import RiskScorer, get_danger_zone_pts
from depth import DepthEstimator
from logger import RiskLogger
from lane import LaneDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Road Accident Predictor")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video file path, '0' for webcam, or phone RTSP URL",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output to outputs/annotated.mp4",
    )
    parser.add_argument(
        "--no-alert", action="store_true", help="Disable audio beep on HIGH risk"
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Show colourised depth map in second window",
    )
    parser.add_argument(
        "--show-edges",
        action="store_true",
        help="Show Canny edge map used for lane detection",
    )
    parser.add_argument(
        "--no-lane",
        action="store_true",
        help="Disable lane detection, use fixed danger zone",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = 0 if args.source == "0" else args.source

    detector = VehicleDetector(
        model_path=config.YOLO_MODEL,
        conf_threshold=config.CONF_THRESHOLD,
    )
    predictor = ObjectPredictor()
    scorer = RiskScorer()
    depth_est = DepthEstimator()
    logger = RiskLogger()
    lane_det = LaneDetector() if (config.LANE_ENABLED and not args.no_lane) else None

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

    cv2.namedWindow("Road Accident Predictor", cv2.WINDOW_NORMAL)
    alert_cooldown = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fh, fw = frame.shape[:2]
            logger.tick()

            if lane_det is not None:
                zone_pts, edge_map, confident = lane_det.detect(frame)
                frame = lane_det.draw_lanes(frame, confident)
            else:
                zone_pts = get_danger_zone_pts(fw, fh)
                edge_map = None
                confident = False

            detections = detector.track(frame)

            predictor.update(detections)

            depth_map = depth_est.estimate(frame)

            for det in detections:
                depth_val = depth_est.get_object_depth(depth_map, det["bbox"])
                predictor.update_depth(det["id"], depth_val)

            risk_scores = scorer.score(detections, predictor, fw, fh, zone_pts=zone_pts)

            logger.log(detections, risk_scores)

            frame = draw_detections(
                frame,
                detections,
                predictor=predictor,
                risk_scores=risk_scores,
                zone_pts=zone_pts,
            )
            frame = draw_hud(frame, risk_scores)

            if lane_det is not None:
                status = "LANE: ON" if confident else "LANE: FALLBACK"
                color = (0, 255, 0) if confident else (0, 165, 255)
                cv2.putText(
                    frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

            cv2.imshow("Road Accident Predictor", frame)

            if args.show_depth:
                depth_vis = depth_est.get_depth_overlay(depth_map)
                cv2.imshow("Depth Map (blue=far, red=close)", depth_vis)

            if args.show_edges and edge_map is not None:
                cv2.imshow("Lane Edges (Canny)", edge_map)

            if writer:
                writer.write(frame)

            if not args.no_alert:
                risks = [v["risk"] for v in risk_scores.values()]
                high_risk = "HIGH" in risks
                medium_risk = "MEDIUM" in risks

                if alert_cooldown == 0:
                    if high_risk:
                        threading.Thread(
                            target=lambda: (
                                winsound.Beep(1200, 200),
                                winsound.Beep(1200, 200),
                            ),
                            daemon=True,
                        ).start()
                        alert_cooldown = config.ALERT_COOLDOWN_FRAMES
                    elif medium_risk:
                        threading.Thread(
                            target=lambda: winsound.Beep(800, 150), daemon=True
                        ).start()
                        alert_cooldown = config.ALERT_COOLDOWN_FRAMES // 2

                if alert_cooldown > 0:
                    alert_cooldown -= 1

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    finally:
        logger.close()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
