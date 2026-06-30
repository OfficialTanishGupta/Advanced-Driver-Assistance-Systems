# Road Accident Predictor

Real-time vehicle detection and tracking system that analyzes traffic movement
to predict collision risk. Built as an intermediate ML/CV portfolio project.

## Current Status: Week 1 — Detection + Tracking

- YOLOv8 object detection (cars, motorcycles, buses, trucks, pedestrians)
- ByteTrack multi-object tracking with persistent IDs

## Roadmap

- [x] Detection + Tracking
- [ ] Kalman filter trajectory prediction
- [ ] Distance estimation heuristic
- [ ] Risk scoring (TTC) + UI overlay

## Setup

\`\`\`bash
pip install -r requirements.txt
python src/main.py --source data/sample_videos/test.mp4
\`\`\`
