"""
utils.py
--------
Shared display utilities: HUD text, status overlays, FPS counter.
"""

import cv2
import time


def put_text(frame, text, pos,
             colour=(255, 255, 255),
             scale: float = 0.65,
             thickness: int = 2):
    """Draw text with a dark shadow for readability on any background."""
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)


def draw_status_badge(frame, label: str, value: str,
                      pos: tuple[int, int],
                      active: bool = False):
    """Draw a small labelled badge (e.g. gesture name)."""
    x, y = pos
    bg = (0, 180, 80) if active else (40, 40, 40)
    text = f"{label}: {value}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + 6), bg, cv2.FILLED)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)


class FPSCounter:
    """Rolling average FPS counter."""

    def __init__(self, window: int = 30):
        self._times  = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0

    def draw(self, frame, pos=(10, 30)):
        fps = self.tick()
        colour = (0, 255, 100) if fps >= 25 else (0, 165, 255)
        put_text(frame, f"FPS: {fps:.1f}", pos, colour=colour, scale=0.6)
