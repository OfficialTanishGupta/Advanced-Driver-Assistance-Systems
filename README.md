# 🚗 Road Accident Predictor
### Real-Time Vehicle Collision Risk Prediction using Computer Vision & Deep Learning

An AI-powered road safety system that predicts potential vehicle collisions in real time using a dashcam, webcam, or Android phone camera. The system combines object detection, multi-object tracking, trajectory prediction, monocular depth estimation, lane detection, and Time-To-Collision (TTC) analysis to estimate collision risk and trigger early warnings.

---

## 📌 Features

- 🚘 Real-time vehicle detection using **YOLOv8**
- 🎯 Multi-object tracking with **ByteTrack**
- 📈 Kalman Filter based trajectory prediction
- ⚠️ Time-To-Collision (TTC) risk estimation
- 📏 Monocular depth estimation using **MiDaS**
- 🛣 Dynamic lane-aware danger zone detection
- 🔊 Audio alerts for high-risk situations
- 📝 CSV logging of collision risk events
- ⚙️ Modular configuration through `config.py`
- 💻 Works with:
  - Dashcam videos
  - Webcam
  - Android IP Webcam
  - Recorded traffic videos

---

# 🏗 Project Architecture

```
Camera / Video
       │
       ▼
YOLOv8 Vehicle Detection
       │
       ▼
ByteTrack Multi-Object Tracking
       │
       ▼
Kalman Trajectory Prediction
       │
       ├───────────────┐
       ▼               ▼
Lane Detection      MiDaS Depth
       │               │
       └──────┬────────┘
              ▼
 Time-To-Collision (TTC)
              │
              ▼
     Risk Score Calculation
              │
      ┌───────┴────────┐
      ▼                ▼
 Audio Alert      CSV Logger
```

---

# ✨ Key Technologies

| Category | Technology |
|----------|------------|
| Programming | Python |
| Object Detection | YOLOv8 |
| Tracking | ByteTrack |
| Trajectory Prediction | Kalman Filter |
| Depth Estimation | MiDaS |
| Lane Detection | OpenCV (Canny + Hough Transform) |
| Risk Estimation | Time-To-Collision (TTC) |
| Visualization | OpenCV |
| Data Logging | CSV |

---

# 📂 Project Structure

```
Road-Accident-Predictor/
│
├── data/
│   ├── sample_videos/
│   └── sample_images/
│
├── models/
│
├── outputs/
│   ├── videos/
│   └── logs/
│
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── depth/
│   ├── lanes/
│   ├── utils/
│   ├── config.py
│   └── main.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Road-Accident-Predictor.git

cd Road-Accident-Predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

### Video File

```bash
python src/main.py --source data/sample_videos/test.mp4
```

### Webcam

```bash
python src/main.py --source 0
```

### Android Phone Camera (IP Webcam)

Install the **IP Webcam** application from the Play Store.

Start the server and note the generated address.

Example

```
http://192.168.1.5:8080/video
```

Run

```bash
python src/main.py --source http://192.168.1.5:8080/video
```

---

# ⚙ Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--source` | Video file, webcam (`0`), or IP camera URL |
| `--save` | Save annotated output video |
| `--no-alert` | Disable audio collision alerts |
| `--show-depth` | Display MiDaS depth estimation window |
| `--show-edges` | Display lane detection edge map |
| `--no-lane` | Disable dynamic lane detection and use a fixed danger zone |

---

# 📊 Collision Risk Pipeline

1. Detect nearby vehicles using YOLOv8.
2. Track each object with ByteTrack.
3. Predict future trajectories using a Kalman Filter.
4. Estimate scene depth using MiDaS.
5. Detect road lanes using Canny Edge Detection and Hough Transform.
6. Calculate Time-To-Collision (TTC).
7. Generate a collision risk score.
8. Trigger audio alerts if the risk exceeds a configurable threshold.
9. Save all risk events into CSV logs.

---

# 📈 Development Roadmap

## ✅ Phase 1
- [x] YOLOv8 Vehicle Detection
- [x] ByteTrack Multi-Object Tracking

## ✅ Phase 2
- [x] Kalman Filter Trajectory Prediction

## ✅ Phase 3
- [x] Time-To-Collision (TTC) Risk Scoring
- [x] Audio Collision Warning

## ✅ Phase 4
- [x] MiDaS Monocular Depth Estimation
- [x] Depth-Based TTC

## ✅ Phase 5
- [x] Dynamic Lane Detection
- [x] Configurable Danger Zone
- [x] CSV Event Logger

---

# 📷 Example Output

The system overlays:

- Bounding boxes
- Vehicle IDs
- Lane boundaries
- Predicted trajectories
- Collision risk score
- TTC value
- Danger zone
- Audio warning status

---

# 🔮 Future Improvements

- [ ] Traffic sign detection
- [ ] Driver drowsiness detection
- [ ] Blind spot monitoring
- [ ] Pedestrian intent prediction
- [ ] Weather-aware collision prediction
- [ ] Multi-camera support
- [ ] GPU optimization
- [ ] Edge deployment using TensorRT
- [ ] ROS2 integration for autonomous vehicles

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project useful:

- ⭐ Star this repository
- 🍴 Fork the project
- 🐞 Report bugs
- 💡 Suggest new features

---

## 👨‍💻 Author

**Tanish Gupta**

AI & Machine Learning Engineer

Passionate about Computer Vision, Autonomous Driving, Artificial Intelligence, and Real-Time Safety Systems.