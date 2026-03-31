"""
ar_canvas.py
------------
Augmented Reality drawing layer overlaid on the live camera feed.

Supported drawing modes
-----------------------
FREEHAND — traces fingertip path as a continuous polyline
LINE      — two-point straight line (click to anchor, release to place)
RECT      — bounding rectangle (click start corner, release end corner)
CIRCLE    — centre + radius (click centre, drag to set radius)
ERASE     — flood-clear on fist gesture

Color palette and brush size are user-selectable via draw_ui().
"""

import cv2
import numpy as np
from enum import Enum, auto


class DrawMode(Enum):
    FREEHAND = auto()
    LINE     = auto()
    RECT     = auto()
    CIRCLE   = auto()


# Default palette
PALETTE = [
    (0,   255,  80),   # green
    (0,   200, 255),   # cyan
    (255,  80,   0),   # blue
    (80,   80, 255),   # red-ish
    (255, 200,   0),   # yellow
    (255, 255, 255),   # white
]

BRUSH_SIZES = [2, 4, 7, 12]


class ARCanvas:
    """
    Maintains a transparent drawing overlay the same size as the camera frame.

    Usage
    -----
    canvas = ARCanvas(w, h)

    # Each frame:
    canvas.draw_ui(frame)           # draw toolbar
    canvas.update(frame, tip, gesture)  # handle drawing logic
    out = canvas.composite(frame)   # blend canvas onto camera frame
    """

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height

        # BGRA canvas (A channel used for compositing)
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)

        self.mode        = DrawMode.FREEHAND
        self.color_idx   = 0
        self.brush_idx   = 1
        self._prev_point = None
        self._anchor     = None          # start point for LINE/RECT/CIRCLE
        self._is_drawing = False

        # UI bar dimensions
        self._bar_h      = 60
        self._swatch_r   = 16
        self._swatch_gap = 48
        self._mode_btn_w = 80

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def color(self):
        return PALETTE[self.color_idx]

    @property
    def brush_size(self):
        return BRUSH_SIZES[self.brush_idx]

    # ── Drawing logic ──────────────────────────────────────────────────────

    def start_stroke(self, point: tuple[int, int]):
        """Called when a pinch/draw gesture begins."""
        self._anchor     = point
        self._prev_point = point
        self._is_drawing = True

    def continue_stroke(self, point: tuple[int, int]):
        """Called every frame while the gesture is held."""
        if not self._is_drawing:
            return
        if self.mode == DrawMode.FREEHAND and self._prev_point:
            cv2.line(self._canvas, self._prev_point, point,
                     self.color, self.brush_size, lineType=cv2.LINE_AA)
            self._prev_point = point

    def end_stroke(self, point: tuple[int, int]):
        """Called when the gesture ends — finalises shape for LINE/RECT/CIRCLE."""
        if not self._is_drawing:
            return
        if self._anchor is None:
            self._is_drawing = False
            return

        if self.mode == DrawMode.LINE:
            cv2.line(self._canvas, self._anchor, point,
                     self.color, self.brush_size, lineType=cv2.LINE_AA)

        elif self.mode == DrawMode.RECT:
            cv2.rectangle(self._canvas, self._anchor, point,
                          self.color, self.brush_size)

        elif self.mode == DrawMode.CIRCLE:
            r = int(np.hypot(point[0] - self._anchor[0],
                             point[1] - self._anchor[1]))
            cv2.circle(self._canvas, self._anchor, r,
                       self.color, self.brush_size, lineType=cv2.LINE_AA)

        self._anchor     = None
        self._prev_point = None
        self._is_drawing = False

    def cancel_stroke(self):
        self._anchor     = None
        self._prev_point = None
        self._is_drawing = False

    def erase_all(self):
        """Clear the entire canvas."""
        self._canvas[:] = 0
        self._anchor     = None
        self._prev_point = None
        self._is_drawing = False

    # ── Preview (ghost shape while dragging) ──────────────────────────────

    def _draw_preview(self, frame: np.ndarray, current: tuple[int, int]):
        """Draw a ghost preview of the shape being dragged."""
        if not self._is_drawing or self._anchor is None:
            return
        col = tuple(max(0, c - 80) for c in self.color)   # dimmer preview

        if self.mode == DrawMode.LINE:
            cv2.line(frame, self._anchor, current, col, 1)

        elif self.mode == DrawMode.RECT:
            cv2.rectangle(frame, self._anchor, current, col, 1)

        elif self.mode == DrawMode.CIRCLE:
            r = int(np.hypot(current[0] - self._anchor[0],
                             current[1] - self._anchor[1]))
            if r > 0:
                cv2.circle(frame, self._anchor, r, col, 1)

    # ── Compositing ───────────────────────────────────────────────────────

    def composite(self, frame: np.ndarray,
                  current_tip=None) -> np.ndarray:
        """
        Blend the canvas onto the camera frame.
        Returns the composited BGR image.
        """
        out = frame.copy()

        # Add canvas where it has been drawn (non-black pixels)
        mask = self._canvas.any(axis=2)
        out[mask] = cv2.addWeighted(frame, 0.15,
                                    self._canvas, 0.85, 0)[mask]

        # Draw ghost preview
        if current_tip:
            self._draw_preview(out, current_tip)

        return out

    # ── Toolbar UI ─────────────────────────────────────────────────────────

    def draw_ui(self, frame: np.ndarray):
        """Draw the colour palette + mode buttons at the top of the frame."""
        # Background bar
        cv2.rectangle(frame, (0, 0), (self.w, self._bar_h),
                      (20, 20, 20), cv2.FILLED)
        cv2.line(frame, (0, self._bar_h), (self.w, self._bar_h),
                 (60, 60, 60), 1)

        # ── Colour swatches ────────────────────────────────────────────────
        for i, col in enumerate(PALETTE):
            cx = 30 + i * self._swatch_gap
            cy = self._bar_h // 2
            cv2.circle(frame, (cx, cy), self._swatch_r, col, cv2.FILLED)
            if i == self.color_idx:
                cv2.circle(frame, (cx, cy), self._swatch_r + 4,
                           (255, 255, 255), 2)

        # ── Mode buttons ───────────────────────────────────────────────────
        modes   = [m.name for m in DrawMode]
        x_start = self.w // 2 - len(modes) * self._mode_btn_w // 2
        for i, name in enumerate(modes):
            x = x_start + i * self._mode_btn_w
            active = (DrawMode[name] == self.mode)
            bg_col = (60, 60, 60) if not active else (0, 180, 80)
            cv2.rectangle(frame, (x, 8), (x + self._mode_btn_w - 6, self._bar_h - 8),
                          bg_col, cv2.FILLED)
            cv2.putText(frame, name, (x + 4, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # ── Brush size indicator ───────────────────────────────────────────
        bx = self.w - 120
        by = self._bar_h // 2
        cv2.putText(frame, f"Brush: {self.brush_size}px",
                    (bx, by + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── Hit-test helpers for UI interaction ────────────────────────────────

    def check_ui_click(self, point: tuple[int, int]) -> bool:
        """
        Check if `point` hits a UI element. If so, handle it and return True.
        Call this before passing the point to the drawing logic.
        """
        px, py = point
        if py > self._bar_h:
            return False   # not in the toolbar

        # Colour swatch hit?
        for i in range(len(PALETTE)):
            cx = 30 + i * self._swatch_gap
            if abs(px - cx) < self._swatch_r + 4:
                self.color_idx = i
                return True

        # Mode button hit?
        modes   = [m for m in DrawMode]
        x_start = self.w // 2 - len(modes) * self._mode_btn_w // 2
        for i, mode in enumerate(modes):
            x = x_start + i * self._mode_btn_w
            if x <= px <= x + self._mode_btn_w - 6:
                self.mode = mode
                return True

        return False

    def cycle_color(self):
        self.color_idx = (self.color_idx + 1) % len(PALETTE)

    def cycle_brush(self):
        self.brush_idx = (self.brush_idx + 1) % len(BRUSH_SIZES)
