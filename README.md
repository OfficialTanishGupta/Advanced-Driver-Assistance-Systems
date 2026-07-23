# Road Accident Predictor

Real-time vehicle collision risk prediction from dashcam or phone camera feed.

## Roadmap
- [x] Week 1 — YOLOv8 detection + ByteTrack tracking
- [x] Week 2 — Kalman filter trajectory prediction
- [x] Week 3 — Danger zone, TTC risk scoring, audio alert
- [x] Depth upgrade — MiDaS monocular depth, depth-based TTC
- [x] Polish — config.py, CSV risk logger
- [x] Lane upgrade — Canny+Hough lane detection, dynamic danger zone


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
| Flag | Description |
|------|-------------|
| `--source` | Video file / `0` webcam / IP stream URL |
| `--save` | Save annotated video to `outputs/` |
| `--no-alert` | Disable audio alerts |
| `--show-depth` | Show depth map in second window |
| `--show-edges` | Show Canny edge map for lane debug |
| `--no-lane` | Use fixed danger zone instead of lane detection |