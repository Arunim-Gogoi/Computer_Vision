# 🖱️ Virtual Mouse + AR Drawing

Control your computer with hand gestures — no hardware required. Uses a standard webcam to detect hand landmarks in real time and translate gestures into OS mouse events and augmented reality drawings overlaid on the live camera feed.

Built as a Computer Vision capstone project using MediaPipe, OpenCV, and PyAutoGUI.

---

## Demo

| Gesture | Action |
|--------|--------|
| ☝️ Index finger moving | Move cursor |
| 🤌 Index + Thumb pinch | Left click |
| 🤌 Middle + Thumb pinch | Right click |
| ✊ Fist | Double click / Erase canvas |
| ✍️ Pinch + drag (Draw mode) | Draw freehand / shapes |

---

## Features

- **Real-time hand tracking** at 21 landmarks via MediaPipe Hands
- **Gesture-to-mouse mapping** with EMA smoothing for stable cursor movement
- **Active zone** — hand only needs to move within a central sub-region of the frame, avoiding edge jitter
- **AR drawing canvas** — freehand, line, rectangle, circle modes
- **Live toolbar** — colour palette + mode switcher rendered in the camera feed
- **FPS counter** and gesture HUD overlay

---

## Project Structure

```
virtual-mouse/
├── main.py              # Entry point — run this
├── hand_tracker.py      # MediaPipe wrapper, landmark extraction, drawing utils
├── gesture_detector.py  # Stateful gesture classifier (pinch, fist detection)
├── mouse_controller.py  # Camera→screen coordinate mapping + PyAutoGUI calls
├── ar_canvas.py         # AR drawing layer (freehand, line, rect, circle)
├── utils.py             # HUD text, FPS counter, status badges
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- A working webcam

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/virtual-mouse.git
cd virtual-mouse

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

> **macOS users:** You must grant Terminal (or your IDE) **Accessibility** and **Camera** permissions in *System Preferences → Privacy & Security* for PyAutoGUI mouse control and webcam access to work.

> **Windows users:** Run as normal user. No admin rights required.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `M` | Toggle between **Mouse** and **Draw** mode |
| `C` | Cycle through draw colours |
| `B` | Cycle brush sizes |
| `E` | Erase entire canvas |
| `Q` | Quit |

---

## How It Works

### 1. Hand Landmark Detection (`hand_tracker.py`)
MediaPipe Hands detects 21 3D landmarks on the hand. The module converts normalised coordinates to pixel space and exposes fingertip positions and a Euclidean distance helper used for all gesture logic.

### 2. Gesture Classification (`gesture_detector.py`)
Gestures are determined purely from landmark geometry — no ML model is trained beyond MediaPipe's built-in detection:

- **Pinch** = Euclidean distance between two fingertips below a threshold (~42px)
- **Fist** = all four fingertip Y-coordinates are below their respective knuckle (MCP) joints
- A frame **cooldown counter** prevents a held pinch from firing multiple clicks per second

### 3. Mouse Control (`mouse_controller.py`)
The index fingertip position is mapped from camera space to screen space using `numpy.interp`, constrained to an **active zone** (central 64% of the frame). An **exponential moving average** (α = 0.25) smooths out per-frame jitter before calling `pyautogui.moveTo()`.

### 4. AR Canvas (`ar_canvas.py`)
A separate `numpy` image (same resolution as the camera frame) acts as the drawing layer. It is composited onto the camera feed using OpenCV's `addWeighted`. Shape previews are rendered directly on the frame (not committed to the canvas) until the gesture is released.

---

## Tuning

Edit the constants at the top of each module to adjust behaviour:

| Parameter | File | Default | Effect |
|-----------|------|---------|--------|
| `PINCH_THRESHOLD` | `gesture_detector.py` | `42 px` | Sensitivity of click/draw gestures |
| `CLICK_COOLDOWN` | `gesture_detector.py` | `12 frames` | Prevents click spam |
| `smoothing` | `mouse_controller.py` | `0.25` | Cursor lag vs stability |
| `margin_frac` | `mouse_controller.py` | `0.18` | Active zone size |

---

## Requirements

```
opencv-python>=4.8.0
mediapipe>=0.10.0
pyautogui>=0.9.54
numpy>=1.24.0
```

---

## Course Context

This project was built for the **Computer Vision** course (BYOP capstone). It applies:
- **Module 1** — Image processing, colour space conversion (BGR → RGB for MediaPipe)
- **Module 3** — Feature extraction (landmark geometry), image segmentation concepts
- **Module 4** — Motion analysis (fingertip tracking across frames), pattern analysis (gesture classification)
- **Module 2** — Single-camera depth cues (relative landmark distances used for 3D gesture inference)

---

## Limitations & Future Work

- Currently supports one hand only
- Pinch threshold may need adjustment per lighting condition
- Scroll gesture not yet implemented (groundwork exists in `mouse_controller.py`)
- Could replace distance-threshold classifier with a trained neural network for more complex gestures

---

## License

MIT
