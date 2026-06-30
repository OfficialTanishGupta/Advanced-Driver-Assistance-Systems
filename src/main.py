import argparse
import cv2

from detector import VehicleDetector
from tracker import draw_detections


def parse_args():
    parser = argparse.ArgumentParser(description="Road Accident Predictor - Week 1")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Path to video file, or '0' for webcam, or phone stream URL",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save annotated output video to outputs/"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = 0 if args.source == "0" else args.source

    detector = VehicleDetector(model_path="yolov8n.pt")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: could not open video source {source}")
        return

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter("outputs/annotated.mp4", fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.track(frame)
        frame = draw_detections(frame, detections)

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
