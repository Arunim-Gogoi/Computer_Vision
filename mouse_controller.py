"""
mouse_controller.py
-------------------
Maps hand landmark positions (camera space) to screen coordinates
and fires OS-level mouse events via PyAutoGUI.

Key design decisions
--------------------
- Only the index fingertip drives the cursor (most natural mapping).
- The active tracking zone is a central sub-region of the camera frame,
  which avoids edge jitter and gives a comfortable range of motion.
- Smoothing (exponential moving average) prevents cursor jitter caused
  by per-frame landmark noise.
"""

import pyautogui
import numpy as np

# PyAutoGUI safety — set to False only after testing
pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0          # Remove built-in delay; we handle timing


class MouseController:
    """
    Converts camera-space fingertip positions to screen mouse events.

    Parameters
    ----------
    cam_w, cam_h   : camera frame resolution
    screen_w/h     : target screen resolution (auto-detected if None)
    smoothing      : EMA factor [0–1]. Higher = smoother but laggier.
    margin_frac    : fraction of the frame to use as the active zone
                     (0.25 means 25% margin on each side → 50% active area)
    """

    def __init__(self,
                 cam_w: int, cam_h: int,
                 screen_w: int = None, screen_h: int = None,
                 smoothing: float = 0.25,
                 margin_frac: float = 0.20):

        self.cam_w = cam_w
        self.cam_h = cam_h

        scr = pyautogui.size()
        self.screen_w = screen_w or scr.width
        self.screen_h = screen_h or scr.height

        self.smoothing    = smoothing
        self.margin_frac  = margin_frac

        # Active zone boundaries in camera space
        self.zone_x1 = int(cam_w * margin_frac)
        self.zone_x2 = int(cam_w * (1 - margin_frac))
        self.zone_y1 = int(cam_h * margin_frac)
        self.zone_y2 = int(cam_h * (1 - margin_frac))

        # Smoothed cursor position (starts at screen centre)
        self._smooth_x = float(self.screen_w  // 2)
        self._smooth_y = float(self.screen_h // 2)

    # ── Public API ─────────────────────────────────────────────────────────

    def move(self, cam_point: tuple[int, int]):
        """
        Move the OS cursor to the screen position corresponding to cam_point.
        Applies active-zone clamping + EMA smoothing.
        """
        cx, cy = cam_point

        # Clamp to active zone
        cx = max(self.zone_x1, min(cx, self.zone_x2))
        cy = max(self.zone_y1, min(cy, self.zone_y2))

        # Map to screen coords
        tx = np.interp(cx, [self.zone_x1, self.zone_x2], [0, self.screen_w])
        ty = np.interp(cy, [self.zone_y1, self.zone_y2], [0, self.screen_h])

        # EMA smoothing
        α = self.smoothing
        self._smooth_x = α * tx + (1 - α) * self._smooth_x
        self._smooth_y = α * ty + (1 - α) * self._smooth_y

        pyautogui.moveTo(int(self._smooth_x), int(self._smooth_y))

    def left_click(self):
        pyautogui.click(button="left")

    def right_click(self):
        pyautogui.click(button="right")

    def double_click(self):
        pyautogui.doubleClick()

    def scroll(self, direction: int = 1):
        """Scroll up (positive) or down (negative)."""
        pyautogui.scroll(direction * 3)

    # ── Zone drawing helper ────────────────────────────────────────────────

    def draw_active_zone(self, frame, colour=(80, 80, 80), thickness=1):
        """Draw the active tracking zone rectangle on the frame."""
        import cv2
        cv2.rectangle(frame,
                      (self.zone_x1, self.zone_y1),
                      (self.zone_x2, self.zone_y2),
                      colour, thickness)
