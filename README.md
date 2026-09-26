# 🧙‍♂️ Invisibility Gesture

A real-time **invisibility cloak** powered by hand gestures. Pinch your thumb and index finger to vanish — no green screen, no manual background capture needed. The system learns the background automatically while you move, then replaces you with it on command.

Built with **OpenCV**, **MediaPipe**, and **NumPy**.

---

## ✨ Features

- **Auto Background Learning** — No need to step out of frame. The system continuously learns the background as you move around using an exponential moving average (EMA).
- **Hand Gesture Toggle** — Pinch your thumb + index finger to toggle invisibility on/off.
- **Person Segmentation** — Uses MediaPipe's Selfie Segmenter to accurately separate you from the background in real time.
- **Soft Blending** — Smooth, feathered edges for a more convincing invisibility effect (no harsh cutouts).
- **Learned-Area Awareness** — Only hides you where the background has been learned, preventing ghosting artifacts.
- **Live HUD** — On-screen overlay showing mode status, background learning progress, and a picture-in-picture preview of the background model.
- **Auto Model Download** — Required ML models are downloaded automatically on first run.

---

## 🛠️ Requirements

- Python 3.8+
- A webcam

### Dependencies

```
opencv-python
mediapipe
numpy
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sharjeelx03/Invisibility-Gesture.git
cd Invisibility-Gesture
```

### 2. Install Dependencies

```bash
pip install opencv-python mediapipe numpy
```

### 3. Run

```bash
python Invisibility-Gesture.py
```

On first launch, the required MediaPipe model files (`selfie_segmenter.tflite` and `hand_landmarker.task`) will be **downloaded automatically** if not already present.

---

## 🎮 Controls

| Key / Gesture | Action |
|---|---|
| **Pinch** (thumb + index) | Toggle invisibility on/off |
| `R` | Reset background model |
| `Q` | Quit |

---

## 📖 How It Works

1. **Background Learning** — Each frame, pixels confidently identified as *not a person* by the segmenter are blended into the background model via EMA. Moving around exposes more of the scene, gradually building a complete background.

2. **Person Segmentation** — MediaPipe's Selfie Segmenter produces a per-pixel confidence mask of where a person is in the frame.

3. **Pinch Detection** — MediaPipe's Hand Landmarker tracks 21 hand landmarks. When the normalized distance between the thumb tip (landmark 4) and index fingertip (landmark 8) drops below a threshold, a pinch is registered.

4. **Invisibility Compositing** — When invisible mode is active, person pixels are replaced with the learned background using soft alpha blending, but *only* where the background has already been learned. This prevents the "frozen person" artifact you'd get from naïvely replacing unlearned regions.

---

## 📁 Project Structure

```
Invisibility-Gesture/
├── Invisibility-Gesture.py      # Main application
├── selfie_segmenter.tflite      # Segmentation model (auto-downloaded)
├── hand_landmarker.task         # Hand tracking model (auto-downloaded)
└── README.md
```

---

## 💡 Tips for Best Results

- **Lighting** — Consistent, even lighting produces the best segmentation and background model.
- **Move around** — Wave your arms, lean side to side. The more of the background you expose, the more complete the model becomes.
- **Wait for 90%+** — The HUD shows background learning progress. Results improve significantly once it passes ~90%.
- **Avoid fast motion** — The EMA-based background model works best with a relatively static scene behind you.
- **Reset if needed** — Press `R` to clear the background model and start over if the scene changes.

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute.
