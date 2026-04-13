# air-writing-pro
🖐️ Gesture-controlled virtual whiteboard using laptop camera — draw, erase, pan &amp; zoom with hand gestures in real-time. Built with Python, OpenCV, MediaPipe &amp; NumPy.

# 🖐️ Air Writing Pro

> A gesture-controlled virtual whiteboard powered by your laptop camera.  
> No stylus. No touchscreen. Just your hand and Python.

![Python](https://img.shields.io/badge/Python-3.9--3.11-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Features

| Gesture | Action |
|---|---|
| ☝️ Index finger only | Draw |
| ✌️ Index + Middle | Erase |
| 🖐️ Open palm | Clear canvas |
| 🤟 4 fingers (no thumb) | Pan canvas |
| 🤲 Both hands pinch | Zoom in / out |
| 👆 Point at toolbar | Select tool / colour |

- 🎨 8 colours + variable brush & eraser sizes  
- ↩️ 50-step undo history  
- 📐 Adaptive precision — slow hand = fine strokes, fast hand = fluid lines  
- 💾 One-key PNG export  
- 📊 Live FPS counter + gesture label HUD  
- 🖱️ Mouse fallback support  

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/abdulhadinp/air-writing-pro.git
cd air-writing-pro

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python air_writing_pro.py
```

> **macOS:** Go to System Settings → Privacy & Security → Camera → enable your terminal app.

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `q` | Quit |
| `c` | Clear canvas |
| `u` | Undo |
| `s` | Save as PNG |
| `+` / `-` | Zoom in / out |
| `0` | Reset view |

---

## 🛠️ Tech Stack

- **Python** 3.9–3.11  
- **OpenCV** — frame capture, rendering, compositing  
- **MediaPipe** — 21-landmark real-time hand tracking  
- **NumPy** — pixel-level drawing engine  

---

## 📁 Project Structure
air-writing-pro/
├── air_writing_pro.py   # Single-file version (run this)
├── multifile/
│   ├── main.py
│   ├── hand_tracker.py
│   ├── gesture_controller.py
│   ├── drawing_canvas.py
│   └── ui_manager.py
├── requirements.txt
└── README.md

---

## 📄 License

MIT — free to use, modify, and distribute.
