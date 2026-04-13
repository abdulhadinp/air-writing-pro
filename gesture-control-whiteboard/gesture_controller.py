"""
gesture_controller.py
─────────────────────
Maps per-frame finger-state dictionaries to high-level Gesture enums.
Uses a majority-vote sliding window for temporal stability (no flicker).
"""

from collections import deque
from enum import Enum, auto
from typing import Dict, Optional, Tuple

# ── How many past frames to include in the majority vote ────
GESTURE_WIN = 7

# ── Minimum votes required before switching gesture ─────────
MIN_VOTES = 3


class Gesture(Enum):
    IDLE     = auto()   # Fist / all fingers curled — no action
    DRAW     = auto()   # Index only up
    ERASE    = auto()   # Index + Middle up
    CLEAR    = auto()   # Open palm (all 5 extended)
    PAN      = auto()   # 4 fingers up, thumb down
    UI_HOVER = auto()   # Index pointing at toolbar zone


class GestureController:
    """
    Converts {finger → bool} dicts produced by HandTracker into
    a smooth, stable Gesture enum value suitable for driving the app.

    Usage
    ─────
        ctrl = GestureController(toolbar_height=80)
        # inside loop:
        gesture = ctrl.update(finger_states, cursor_position)
    """

    def __init__(self, toolbar_height: int = 80, ui_margin: int = 24):
        self._toolbar_h = toolbar_height
        self._ui_margin = ui_margin
        self._hist: deque = deque(maxlen=GESTURE_WIN)
        self.current: Gesture = Gesture.IDLE

    # ── Private rule table ───────────────────────────────────

    def _classify(
        self,
        states: Optional[Dict[str, bool]],
        cursor: Optional[Tuple[int, int]],
    ) -> Gesture:
        """
        Pure, stateless classification for a single frame.
        Priority order prevents ambiguity between overlapping patterns.
        """
        if states is None:
            return Gesture.IDLE

        t = states.get("thumb",  False)
        i = states.get("index",  False)
        m = states.get("middle", False)
        r = states.get("ring",   False)
        p = states.get("pinky",  False)

        # ── Priority table ────────────────────────────────────
        # 1. Open palm → CLEAR
        if t and i and m and r and p:
            return Gesture.CLEAR

        # 2. Four fingers (no thumb) → PAN
        if (not t) and i and m and r and p:
            return Gesture.PAN

        in_ui_zone = cursor is not None and cursor[1] < (self._toolbar_h + self._ui_margin)
        aux_open = int(r) + int(p)
        if in_ui_zone and i and aux_open <= 1:
            return Gesture.UI_HOVER

        # 3. Index + Middle mostly open → ERASE
        if i and m and aux_open <= 1:
            return Gesture.ERASE

        # 4. Index open and others mostly down → DRAW
        if i and aux_open <= 1 and not m:
            return Gesture.DRAW

        # 5. Everything else → IDLE (fist, partial grips, etc.)
        return Gesture.IDLE

    # ── Public ───────────────────────────────────────────────

    def update(
        self,
        states: Optional[Dict[str, bool]],
        cursor: Optional[Tuple[int, int]],
    ) -> Gesture:
        """
        Call once per frame.
        Returns the temporally-stabilised current gesture.
        """
        raw = self._classify(states, cursor)
        self._hist.append(raw)

        if len(self._hist) >= MIN_VOTES:
            # Majority vote over recent history
            counts: Dict[Gesture, int] = {}
            for g in self._hist:
                counts[g] = counts.get(g, 0) + 1
            winner = max(counts, key=lambda g: counts[g])
            self.current = winner

        return self.current

    def reset(self):
        """Clear history and return to IDLE (call on mode changes)."""
        self._hist.clear()
        self.current = Gesture.IDLE
