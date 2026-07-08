# Road Accident Predictor

Real-time vehicle collision risk prediction from dashcam or phone camera feed.

## Roadmap

- [x] Week 1 — YOLOv8 detection + ByteTrack multi-object tracking
- [x] Week 2 — Per-vehicle Kalman filter, trajectory trail, predicted path arrow
- [x] Week 3 — Danger zone trapezoid, heuristic TTC, risk scoring, audio alert

## Setup

\`\`\`bash
pip install -r requirements.txt
python src/main.py --source data/sample_videos/test.mp4
\`\`\`

## Phone camera (live)

1. Install **IP Webcam** app on Android
2. Start server in app, note the IP (e.g. 192.168.1.5:8080)
3. Run:
   \`\`\`bash
   python src/main.py --source http://192.168.1.5:8080/video
   \`\`\`

## Args

| Flag         | Description                             |
| ------------ | --------------------------------------- |
| `--source`   | Video file / webcam `0` / IP stream URL |
| `--save`     | Save annotated video to `outputs/`      |
| `--no-alert` | Disable beep on HIGH risk               |
