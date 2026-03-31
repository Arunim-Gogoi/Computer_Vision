"""
hand_tracker.py
---------------
Core hand tracking module using MediaPipe Hands.
Detects 21 landmarks per hand and exposes utilities for
fingertip positions, Euclidean distances, and frame drawing.

MediaPipe Landmark Index Reference:
    0  WRIST           4  THUMB_TIP
    5  INDEX_MCP       8  INDEX_TIP
    9  MIDDLE_MCP     12  MIDDLE_TIP
   13  RING_MCP       16  RING_TIP
   17  PINKY_MCP      20  PINKY_TIP
"""

import cv2
import mediapipe as mp
import numpy as np

# ── Landmark indices ───────────────────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4
INDEX_MCP  = 5
INDEX_TIP  = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP   = 16
PINKY_TIP  = 20

ALL_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]


class HandTracker:
    """
    Wraps MediaPipe Hands for per-frame landmark detection.

    Parameters
    ----------
    max_hands      : int   — max hands to detect (default 1)
    detection_conf : float — min detection confidence
    tracking_conf  : float — min tracking confidence
    """

    def __init__(self, max_hands: int = 1,
                 detection_conf: float = 0.8,
                 tracking_conf: float = 0.8):

        self._mp_hands  = mp.solutions.hands
        self._mp_draw   = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode        = False,
            max_num_hands            = max_hands,
            min_detection_confidence = detection_conf,
            min_tracking_confidence  = tracking_conf,
        )

        # Public state updated on every process_frame() call
        self.landmarks: list[tuple[int, int]] = []
        self.results = None

    # ── Core ───────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run hand detection on a BGR frame.
        Updates self.landmarks with pixel coords for all 21 points.
        Returns the annotated frame.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self._hands.process(rgb)
        rgb.flags.writeable = True

        self.landmarks = []

        if self.results.multi_hand_landmarks:
            hand_lms = self.results.multi_hand_landmarks[0]

            # Draw skeleton
            self._mp_draw.draw_landmarks(
                frame, hand_lms,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

            # Convert to pixel coords
            for lm in hand_lms.landmark:
                self.landmarks.append((int(lm.x * w), int(lm.y * h)))

            # Highlight fingertips
            for idx in ALL_TIPS:
                cx, cy = self.landmarks[idx]
                cv2.circle(frame, (cx, cy), 10, (0, 255, 180), cv2.FILLED)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)

        return frame

    # ── Accessors ──────────────────────────────────────────────────────────

    def get(self, index: int) -> tuple[int, int] | None:
        """Return pixel (x, y) of landmark index, or None if not detected."""
        return self.landmarks[index] if len(self.landmarks) > index else None

    def get_fingertips(self) -> dict[str, tuple[int, int] | None]:
        """Named fingertip positions."""
        return {
            "thumb":  self.get(THUMB_TIP),
            "index":  self.get(INDEX_TIP),
            "middle": self.get(MIDDLE_TIP),
            "ring":   self.get(RING_TIP),
            "pinky":  self.get(PINKY_TIP),
        }

    def hand_detected(self) -> bool:
        return len(self.landmarks) == 21

    # ── Geometry ───────────────────────────────────────────────────────────

    @staticmethod
    def distance(p1, p2) -> float:
        """Euclidean distance between two (x, y) points."""
        if p1 is None or p2 is None:
            return float("inf")
        return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

    # ── Drawing helpers ────────────────────────────────────────────────────

    def draw_line(self, frame, p1, p2, colour=(0, 255, 100), show_dist=True):
        """Draw a line between two points, optionally showing distance."""
        if p1 is None or p2 is None:
            return
        cv2.line(frame, p1, p2, colour, 2)
        if show_dist:
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            dist = int(self.distance(p1, p2))
            cv2.putText(frame, f"{dist}px", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    def draw_pinch_dot(self, frame, p1, p2, colour):
        """Draw a dot at the midpoint of a pinch gesture."""
        if p1 and p2:
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.circle(frame, mid, 14, colour, cv2.FILLED)

    def release(self):
        self._hands.close()
