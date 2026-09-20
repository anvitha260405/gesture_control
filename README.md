# Gesture-Controlled Presentation System

A real-time hand-gesture recognition system that lets users control slide presentations — navigating forward/backward and zooming in/out — without touching a keyboard or mouse.

Built as a team project during a hackathon.

## Team
- Ch Anvitha
- Mancha Laxmi Anusha

## Overview
The system uses a webcam feed to detect hand landmarks in real time and maps specific gestures to presentation controls — next/previous slide and zoom. It removes the need for a physical clicker or keyboard, making presentations more interactive and hands-free.

## Features
- Real-time hand tracking via webcam
- Gesture-based slide navigation (next / previous)
- Gesture-based zoom control
- ~80% gesture recognition accuracy in real-time testing

## Tech Stack
- **Python** — core application logic
- **OpenCV** — video capture and image processing
- **MediaPipe** — hand landmark detection and tracking

## File Structure
| File | Purpose |
|---|---|
| `gesture_control.py` | Main entry point — captures webcam feed and routes detected gestures to actions |
| `air_play.py` | Handles slide play/navigation gestures |
| `air_screen.py` | Screen/display control logic |
| `zoom.py` | Zoom-in/zoom-out gesture handling |

## How It Works
1. Webcam captures live video feed.
2. MediaPipe detects hand landmarks in each frame.
3. Landmark positions are interpreted into gestures (e.g., swipe left/right, pinch).
4. Recognized gestures trigger the corresponding presentation action (next slide, previous slide, zoom).

## Run Locally
```bash
pip install opencv-python mediapipe
python gesture_control.py
```

## Future Improvements
- Improve accuracy beyond 80% with gesture smoothing/debouncing
- Add more gesture types (e.g., laser pointer, annotation mode)
- Support for multiple camera angles/lighting conditions
