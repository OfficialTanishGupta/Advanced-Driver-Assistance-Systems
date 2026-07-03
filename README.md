# Road Accident Predictor

Real-time vehicle detection and tracking system that analyzes traffic movement
to predict collision risk. Built as an intermediate ML/CV portfolio project.

## Current Status: Week 1 — Detection + Tracking

- YOLOv8 object detection (cars, motorcycles, buses, trucks, pedestrians)
- ByteTrack multi-object tracking with persistent IDs

## Current Status: Week 2 — Kalman Filter Trajectory Prediction
- YOLOv8 object detection (cars, motorcycles, buses, trucks, pedestrians)
- ByteTrack multi-object tracking with persistent IDs
- Per-vehicle Kalman filter (constant-velocity model, state: x y vx vy)
- Trail visualization of past positions
- Predicted future trajectory arrow (~1 second ahead)

## Roadmap
- [x] Detection + Tracking
- [x] Kalman filter trajectory prediction
- [ ] Distance estimation heuristic
- [ ] Risk scoring (TTC) + danger zone
- [ ] Audio alert + final UI polish

## Setup

\`\`\`bash
pip install -r requirements.txt
python src/main.py --source data/sample_videos/test.mp4
\`\`\`
