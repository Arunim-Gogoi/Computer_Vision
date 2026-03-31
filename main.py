"""
main.py
-------
Virtual Mouse with AR Drawing — entry point.

Modes
-----
MOUSE  (default) — hand controls the OS cursor; pinches fire clicks
DRAW             — AR canvas activated; draw shapes with gestures

Toggle modes with  M  key.

Keyboard shortcuts
------------------
  M   — toggle Mouse ↔ Draw mode
  C   — cycle draw colour
  B   — cycle brush size
  E   — erase canvas
  Q   — quit

Gesture map
-----------
  Index tip moving          → move cursor (MOUSE) / draw (DRAW/FREEHAND)
  Index + Thumb pinch       → left click (MOUSE) / anchor shape (DRAW)
  Middle + Thumb pinch      → right click (MOUSE) / erase (DRAW)
  Fist                      → erase canvas (DRAW mode)
"""

import cv2
import pyautogui

from hand_tracker     import HandTracker, INDEX_TIP, THUMB_TIP, MIDDLE_TIP
from gesture_detector import GestureDetector, Gesture
from mouse_controller import MouseController
from ar_canvas        import ARCanvas, DrawMode
from utils            import FPSCounter, put_text, draw_status_badge


# ── Config ─────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
FRAME_W        = 1280
FRAME_H        = 720
FLIP           = True     # mirror frame (selfie-style)
SHOW_LANDMARKS = True     # toggle landmark overlay

APP_MODE_MOUSE = "MOUSE"
APP_MODE_DRAW  = "DRAW"


def main():
    # ── Init camera ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Virtual Mouse] Camera: {actual_w}x{actual_h}")
    print(f"[Virtual Mouse] Screen: {pyautogui.size()}")
    print("[Virtual Mouse] Press M to toggle mode, Q to quit.\n")

    # ── Init modules ───────────────────────────────────────────────────────
    tracker   = HandTracker(max_hands=1, detection_conf=0.8, tracking_conf=0.8)
    detector  = GestureDetector(pinch_threshold=42, click_cooldown=12)
    mouse     = MouseController(actual_w, actual_h, smoothing=0.25, margin_frac=0.18)
    canvas    = ARCanvas(actual_w, actual_h)
    fps       = FPSCounter()

    app_mode  = APP_MODE_MOUSE
    drawing   = False        # True while left-pinch is held in DRAW mode
    prev_tip  = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Camera read failed.")
            break

        if FLIP:
            frame = cv2.flip(frame, 1)

        # ── Hand tracking ──────────────────────────────────────────────────
        frame = tracker.process_frame(frame)

        gesture   = detector.detect(tracker)
        index_tip = tracker.get(INDEX_TIP)
        thumb_tip = tracker.get(THUMB_TIP)
        mid_tip   = tracker.get(MIDDLE_TIP)

        # ── Draw toolbar ───────────────────────────────────────────────────
        canvas.draw_ui(frame)

        # ── Mode logic ─────────────────────────────────────────────────────
        if app_mode == APP_MODE_MOUSE:
            _handle_mouse_mode(frame, tracker, detector, mouse,
                               gesture, index_tip, thumb_tip, mid_tip)

        else:  # DRAW mode
            drawing = _handle_draw_mode(frame, tracker, detector, canvas,
                                        gesture, index_tip, thumb_tip,
                                        prev_tip, drawing)

        prev_tip = index_tip

        # ── Composite AR canvas ────────────────────────────────────────────
        frame = canvas.composite(
            frame,
            current_tip=index_tip if (app_mode == APP_MODE_DRAW and drawing) else None
        )

        # ── Active-zone guide (mouse mode) ─────────────────────────────────
        if app_mode == APP_MODE_MOUSE:
            mouse.draw_active_zone(frame)

        # ── HUD ────────────────────────────────────────────────────────────
        fps.draw(frame, pos=(10, actual_h - 20))

        mode_col = (0, 200, 255) if app_mode == APP_MODE_MOUSE else (0, 200, 80)
        draw_status_badge(frame, "MODE", app_mode,
                          pos=(10, actual_h - 50), active=True)

        if tracker.hand_detected():
            gest_name = gesture.name if gesture else "–"
            draw_status_badge(frame, "GESTURE", gest_name,
                              pos=(200, actual_h - 50),
                              active=(gesture not in (Gesture.NONE, Gesture.MOVE)))

        _draw_legend(frame, actual_h)

        cv2.imshow("Virtual Mouse + AR Drawing", frame)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            app_mode = APP_MODE_DRAW if app_mode == APP_MODE_MOUSE else APP_MODE_MOUSE
            canvas.cancel_stroke()
            print(f"[Mode] Switched to {app_mode}")
        elif key == ord("c"):
            canvas.cycle_color()
        elif key == ord("b"):
            canvas.cycle_brush()
        elif key == ord("e"):
            canvas.erase_all()

    # ── Cleanup ────────────────────────────────────────────────────────────
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[Virtual Mouse] Exited cleanly.")


# ── Mode handlers ─────────────────────────────────────────────────────────

def _handle_mouse_mode(frame, tracker, detector, mouse,
                       gesture, index_tip, thumb_tip, mid_tip):
    """Drive the OS mouse from hand position and gestures."""
    if index_tip is None:
        return

    # Always move cursor to index fingertip
    mouse.move(index_tip)

    if gesture == Gesture.LEFT_CLICK:
        mouse.left_click()
        # Visual flash
        tracker.draw_pinch_dot(frame, index_tip, thumb_tip, (0, 255, 100))

    elif gesture == Gesture.RIGHT_CLICK:
        mouse.right_click()
        tracker.draw_pinch_dot(frame, mid_tip, thumb_tip, (0, 80, 255))

    elif gesture == Gesture.ERASE:
        # Fist in mouse mode → double-click
        mouse.double_click()

    # Show distance lines for feedback
    tracker.draw_line(frame, index_tip, thumb_tip, (0, 255, 100))
    tracker.draw_line(frame, mid_tip,   thumb_tip, (0, 80, 255))


def _handle_draw_mode(frame, tracker, detector, canvas,
                      gesture, index_tip, thumb_tip,
                      prev_tip, currently_drawing) -> bool:
    """Handle AR canvas drawing gestures. Returns updated drawing state."""
    if index_tip is None:
        if currently_drawing:
            canvas.cancel_stroke()
        return False

    # Check if tap is on the UI toolbar
    ui_hit = canvas.check_ui_click(index_tip)

    if gesture == Gesture.ERASE or detector.is_right_pinching(tracker):
        canvas.erase_all()
        return False

    left_pinch = detector.is_left_pinching(tracker)

    if left_pinch and not ui_hit:
        if not currently_drawing:
            canvas.start_stroke(index_tip)
            return True
        else:
            canvas.continue_stroke(index_tip)
            return True
    else:
        if currently_drawing:
            canvas.end_stroke(index_tip if index_tip else prev_tip)
        return False


# ── Legend ─────────────────────────────────────────────────────────────────

def _draw_legend(frame, frame_h):
    lines = [
        "M: toggle mode   C: colour   B: brush   E: erase   Q: quit",
        "MOUSE — Index+Thumb=LClick  Middle+Thumb=RClick",
        "DRAW  — Pinch to draw  |  Fist / Right-pinch = erase",
    ]
    y = frame_h - 110
    for line in lines:
        put_text(frame, line, (10, y), colour=(160, 160, 160), scale=0.45, thickness=1)
        y += 20


if __name__ == "__main__":
    main()
