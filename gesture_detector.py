"""
gesture_detector.py
-------------------
Translates raw landmark positions into discrete gesture events.

Gestures Detected
-----------------
MOVE        — index finger extended, no pinch
LEFT_CLICK  — index tip ↔ thumb tip distance < threshold  (pinch)
RIGHT_CLICK — middle tip ↔ thumb tip distance < threshold (pinch)
DRAW        — index tip moving while in draw mode
ERASE       — fist (all fingers curled)

Design note: click events use a small cooldown counter so a held
pinch doesn't fire hundreds of clicks per second.
"""

from enum import Enum, auto
from hand_tracker import HandTracker, INDEX_TIP, MIDDLE_TIP, THUMB_TIP, RING_TIP, PINKY_TIP, INDEX_MCP


class Gesture(Enum):
    NONE        = auto()
    MOVE        = auto()
    LEFT_CLICK  = auto()
    RIGHT_CLICK = auto()
    DRAW        = auto()
    ERASE       = auto()
    SCROLL_UP   = auto()
    SCROLL_DOWN = auto()


# ── Tunable thresholds ─────────────────────────────────────────────────────
PINCH_THRESHOLD  = 42   # px — distance below which a pinch is detected
FIST_THRESHOLD   = 70   # px — all tips must be within this of the palm centre
CLICK_COOLDOWN   = 12   # frames — prevent click spam while holding a pinch


class GestureDetector:
    """
    Stateful gesture classifier.
    Call detect() on every frame with the current HandTracker instance.
    """

    def __init__(self,
                 pinch_threshold: int = PINCH_THRESHOLD,
                 click_cooldown:  int = CLICK_COOLDOWN):

        self.pinch_threshold = pinch_threshold
        self.click_cooldown  = click_cooldown

        self._lclick_cooldown = 0
        self._rclick_cooldown = 0
        self._prev_gesture    = Gesture.NONE

    # ── Main API ───────────────────────────────────────────────────────────

    def detect(self, tracker: HandTracker) -> Gesture:
        """Return the current Gesture for this frame."""
        if not tracker.hand_detected():
            return Gesture.NONE

        # Tick cooldowns
        if self._lclick_cooldown > 0:
            self._lclick_cooldown -= 1
        if self._rclick_cooldown > 0:
            self._rclick_cooldown -= 1

        index  = tracker.get(INDEX_TIP)
        middle = tracker.get(MIDDLE_TIP)
        thumb  = tracker.get(THUMB_TIP)
        ring   = tracker.get(RING_TIP)
        pinky  = tracker.get(PINKY_TIP)

        left_dist  = tracker.distance(index, thumb)
        right_dist = tracker.distance(middle, thumb)

        # ── Fist = erase ──────────────────────────────────────────────────
        if self._is_fist(tracker):
            return Gesture.ERASE

        # ── Left click (index + thumb pinch) ─────────────────────────────
        if left_dist < self.pinch_threshold and self._lclick_cooldown == 0:
            self._lclick_cooldown = self.click_cooldown
            return Gesture.LEFT_CLICK

        # ── Right click (middle + thumb pinch) ───────────────────────────
        if right_dist < self.pinch_threshold and self._rclick_cooldown == 0:
            self._rclick_cooldown = self.click_cooldown
            return Gesture.RIGHT_CLICK

        # ── Move / Draw (index extended, no pinch) ───────────────────────
        return Gesture.MOVE

    def is_left_pinching(self, tracker: HandTracker) -> bool:
        """True while index+thumb are close (sustained pinch, not single event)."""
        index = tracker.get(INDEX_TIP)
        thumb = tracker.get(THUMB_TIP)
        return tracker.distance(index, thumb) < self.pinch_threshold

    def is_right_pinching(self, tracker: HandTracker) -> bool:
        """True while middle+thumb are close (sustained pinch)."""
        middle = tracker.get(MIDDLE_TIP)
        thumb  = tracker.get(THUMB_TIP)
        return tracker.distance(middle, thumb) < self.pinch_threshold

    # ── Helpers ────────────────────────────────────────────────────────────

    def _is_fist(self, tracker: HandTracker) -> bool:
        """
        Rough fist detection: all four fingertips are below their MCP joints
        (i.e., fingers are curled inward).
        """
        if not tracker.hand_detected():
            return False

        lms = tracker.landmarks

        def tip_below_mcp(tip_idx, mcp_idx):
            # In image coordinates, y increases downward
            # Tip below MCP means tip_y > mcp_y
            return lms[tip_idx][1] > lms[mcp_idx][1]

        return (
            tip_below_mcp(INDEX_TIP,  5)  and
            tip_below_mcp(MIDDLE_TIP, 9)  and
            tip_below_mcp(RING_TIP,   13) and
            tip_below_mcp(PINKY_TIP,  17)
        )
