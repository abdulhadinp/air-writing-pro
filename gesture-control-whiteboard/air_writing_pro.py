#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════╗
║          Air Writing Pro — Gesture-Controlled Whiteboard      ║
║          Single-File Production Version                       ║
║                                                               ║
║  Gestures:                                                    ║
║    Index only          → Draw                                 ║
║    Index + Middle      → Erase                                ║
║    Open palm (all 5)   → Clear canvas                         ║
║    4 fingers (no thumb)→ Pan canvas                           ║
║    Finger on toolbar   → Select tool / color / action         ║
║                                                               ║
║  Keys: q=quit  c=clear  u=undo  s=save                        ║
╚═══════════════════════════════════════════════════════════════╝
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import urllib.request
from collections import deque
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

_mp_solutions = getattr(mp, "solutions", None)


# ══════════════════════════════════════════════════════════════
#  ENUMERATIONS
# ══════════════════════════════════════════════════════════════

class Gesture(Enum):
    IDLE     = auto()   # Fist / all down
    DRAW     = auto()   # Index only → draw
    ERASE    = auto()   # Index + Middle → erase
    CLEAR    = auto()   # Open palm (all 5)
    PAN      = auto()   # 4 fingers (no thumb)
    UI_HOVER = auto()   # Index pointing at toolbar


class Tool(Enum):
    PEN    = "pen"
    ERASER = "eraser"


class GestureInputMode(Enum):
    ONE_HAND = "1h"
    TWO_HAND = "2h"


# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

TOOLBAR_H      = 80       # pixels
HOVER_TIME     = 0.55     # seconds dwell to activate button
UI_ACTIVE_MARGIN = 24     # allow a small buffer below the toolbar for easier selection

PALETTE: Dict[str, Tuple[int, int, int]] = {   # BGR
    "White":   (255, 255, 255),
    "Red":     (40,  40,  220),
    "Blue":    (210, 80,  20),
    "Green":   (40,  200, 40),
    "Yellow":  (0,   215, 215),
    "Cyan":    (210, 210, 0),
    "Magenta": (195, 40,  195),
    "Orange":  (0,   140, 255),
}

BRUSH_SIZES  = [3, 6, 10, 16, 24]
ERASER_SIZES = [20, 35, 55, 75]

CURSOR_ALPHA_SLOW = 0.20   # stronger smoothing for tiny movements
CURSOR_ALPHA_FAST = 0.72   # faster response for larger movements
CURSOR_FAST_PX    = 35.0   # movement (px/frame) where smoothing becomes responsive
STROKE_SMOOTH = 5     # Moving-avg window for stroke points
MAX_UNDO      = 50    # Max undo states kept in RAM
GESTURE_WIN   = 7     # Majority-vote window for gesture stability
CLEAR_COOLDOWN = 2.5  # seconds between successive clears
DUAL_UNDO_COOLDOWN = 1.0
HAND_LANDMARKER_ENV = "MP_HAND_LANDMARKER_TASK"
HAND_LANDMARKER_URLS = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
)


# ══════════════════════════════════════════════════════════════
#  HAND TRACKER
# ══════════════════════════════════════════════════════════════

class HandTracker:
    """
    Thin wrapper around MediaPipe Hands.
    Provides per-finger up/down state and pixel-space landmark positions.
    """

    # Landmark IDs (MediaPipe canonical ordering)
    WRIST      = 0
    THUMB_IP   = 3;  THUMB_TIP  = 4
    INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_TIP  = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10; MIDDLE_TIP = 12
    RING_PIP   = 14; RING_TIP   = 16
    PINKY_PIP  = 18; PINKY_TIP  = 20

    def __init__(self):
        if _mp_solutions is not None:
            _hands_mod = _mp_solutions.hands
            self._hands = _hands_mod.Hands(
                static_image_mode        = False,
                max_num_hands            = 2,
                model_complexity         = 0,        # 0 = speed, 1 = accuracy
                min_detection_confidence = 0.72,
                min_tracking_confidence  = 0.62,
            )
            self._mp_hands   = _hands_mod
            self._draw_utils = _mp_solutions.drawing_utils
            self._connections = _hands_mod.HAND_CONNECTIONS
        else:
            try:
                self._hands = mp.tasks.vision.HandLandmarker.create_from_options(
                    mp.tasks.vision.HandLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(
                            model_asset_path=self._find_hand_landmarker_task(),
                            delegate=mp.tasks.BaseOptions.Delegate.CPU,
                        ),
                        running_mode=mp.tasks.vision.RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.72,
                        min_hand_presence_confidence=0.62,
                        min_tracking_confidence=0.62,
                    )
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Failed to initialize MediaPipe HandLandmarker. "
                    "Ensure you run from a normal desktop session (not headless/SSH), "
                    "then retry."
                ) from exc
            self._mp_hands   = mp.tasks.vision.HandLandmarksConnections
            self._draw_utils = mp.tasks.vision.drawing_utils
            self._connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        self._hand_landmarks: List = []
        self._hand_labels: List[str] = []
        self.landmarks = None
        self.detected = False

    def _find_hand_landmarker_task(self) -> str:
        env_override = os.getenv(HAND_LANDMARKER_ENV)
        candidates: List[str] = []
        if env_override:
            candidates.append(os.path.expanduser(env_override))
        candidates.extend([
            os.path.join(os.path.dirname(__file__), "hand_landmarker.task"),
            os.path.join(os.getcwd(), "hand_landmarker.task"),
        ])
        candidates = list(dict.fromkeys(candidates))
        for path in candidates:
            if os.path.isfile(path):
                return path

        last_error: Optional[Exception] = None
        for out_path in candidates:
            out_dir = os.path.dirname(out_path) or "."
            if not os.path.isdir(out_dir) or not os.access(out_dir, os.W_OK):
                continue
            for url in HAND_LANDMARKER_URLS:
                try:
                    urllib.request.urlretrieve(url, out_path)
                except Exception as exc:
                    last_error = exc
                    continue
                if os.path.isfile(out_path):
                    return out_path

        tried = "\n  - ".join(candidates)
        detail = f"\nLast download error: {last_error}" if last_error else ""
        raise FileNotFoundError(
            "MediaPipe hand_landmarker.task asset not found.\n"
            f"Tried:\n  - {tried}\n"
            f"Set {HAND_LANDMARKER_ENV} to a local model path or download manually:\n"
            f"  - {HAND_LANDMARKER_URLS[0]}\n"
            f"  - {HAND_LANDMARKER_URLS[1]}"
            f"{detail}"
        )

    # ── Core ─────────────────────────────────────────────────

    def process(self, rgb_frame: np.ndarray):
        """Run MediaPipe inference. Must be called every frame."""
        self._hand_landmarks.clear()
        self._hand_labels.clear()

        if _mp_solutions is not None:
            res = self._hands.process(rgb_frame)
            lms = res.multi_hand_landmarks or []
            handed = res.multi_handedness or []
            for i, hand_lm in enumerate(lms):
                lm = getattr(hand_lm, "landmark", hand_lm)
                self._hand_landmarks.append(lm)
                hobj = handed[i] if i < len(handed) else None
                self._hand_labels.append(self._extract_handedness_label(hobj))
        else:
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            res = self._hands.detect(image)
            lms = res.hand_landmarks if (res and res.hand_landmarks) else []
            handed = res.handedness if (res and res.handedness) else []
            for i, hand_lm in enumerate(lms):
                lm = getattr(hand_lm, "landmark", hand_lm)
                self._hand_landmarks.append(lm)
                hobj = handed[i] if i < len(handed) else None
                self._hand_labels.append(self._extract_handedness_label(hobj))

        self.detected = bool(self._hand_landmarks)
        self.landmarks = self._hand_landmarks[0] if self.detected else None
        return self.landmarks

    def _extract_handedness_label(self, handed_obj) -> str:
        if handed_obj is None:
            return "Unknown"
        cls = getattr(handed_obj, "classification", None)
        if cls and len(cls):
            return str(getattr(cls[0], "label", "Unknown"))
        if isinstance(handed_obj, (list, tuple)) and handed_obj:
            cat = handed_obj[0]
            return str(
                getattr(cat, "category_name", None)
                or getattr(cat, "display_name", None)
                or "Unknown"
            )
        return str(
            getattr(handed_obj, "category_name", None)
            or getattr(handed_obj, "label", None)
            or "Unknown"
        )

    def _normalized_landmarks(self, hand_idx: int = 0):
        if hand_idx < 0 or hand_idx >= len(self._hand_landmarks):
            return None
        return self._hand_landmarks[hand_idx]

    def _finger_states_for(self, lm) -> Dict[str, bool]:
        # Thumb: compare x rather than y (it bends sideways)
        idx_mcp_x = lm[self.INDEX_MCP].x
        wrist_x   = lm[self.WRIST].x
        tt = lm[self.THUMB_TIP]
        ti = lm[self.THUMB_IP]
        if idx_mcp_x > wrist_x:      # right-hand geometry
            thumb_up = tt.x < ti.x
        else:                        # left-hand geometry
            thumb_up = tt.x > ti.x

        hand_scale = max(
            1e-4,
            math.hypot(
                lm[self.INDEX_MCP].x - lm[self.PINKY_PIP].x,
                lm[self.INDEX_MCP].y - lm[self.PINKY_PIP].y,
            ),
        )

        def ext_score(tip_id: int, pip_id: int) -> float:
            return (lm[pip_id].y - lm[tip_id].y) / hand_scale

        idx_s = ext_score(self.INDEX_TIP, self.INDEX_PIP)
        mid_s = ext_score(self.MIDDLE_TIP, self.MIDDLE_PIP)
        rng_s = ext_score(self.RING_TIP, self.RING_PIP)
        pky_s = ext_score(self.PINKY_TIP, self.PINKY_PIP)

        return {
            "thumb":  thumb_up,
            "index":  idx_s > -0.08,
            "middle": mid_s > -0.03,
            "ring":   rng_s >  0.02,
            "pinky":  pky_s >  0.02,
        }

    def hand_infos(self, fw: int, fh: int) -> List[Dict[str, object]]:
        infos: List[Dict[str, object]] = []
        for i, lm in enumerate(self._hand_landmarks):
            idx = self.tip_px(self.INDEX_TIP, fw, fh, hand_idx=i)
            wrist = self.tip_px(self.WRIST, fw, fh, hand_idx=i)
            mmcp = self.tip_px(self.MIDDLE_MCP, fw, fh, hand_idx=i)
            if wrist and mmcp:
                center = ((wrist[0] + mmcp[0]) // 2, (wrist[1] + mmcp[1]) // 2)
            else:
                center = idx or (0, 0)
            infos.append({
                "index": i,
                "label": self._hand_labels[i] if i < len(self._hand_labels) else "Unknown",
                "states": self._finger_states_for(lm),
                "index_tip": idx,
                "center": center,
                "landmarks": lm,
            })
        return infos

    # ── Finger states ────────────────────────────────────────

    def finger_states(self, fw: int, fh: int) -> Optional[Dict[str, bool]]:
        """
        Returns {'thumb','index','middle','ring','pinky'} → bool (extended?).
        Returns None when no hand is detected.
        """
        lm = self._normalized_landmarks()
        if not lm:
            return None

        return self._finger_states_for(lm)

    # ── Positions ────────────────────────────────────────────

    def tip_px(self, tip_id: int, fw: int, fh: int, hand_idx: int = 0) -> Optional[Tuple[int, int]]:
        lm = self._normalized_landmarks(hand_idx)
        if not lm:
            return None
        lm = lm[tip_id]
        return int(lm.x * fw), int(lm.y * fh)

    def index_tip(self, fw: int, fh: int, hand_idx: int = 0) -> Optional[Tuple[int, int]]:
        return self.tip_px(self.INDEX_TIP, fw, fh, hand_idx=hand_idx)

    def middle_tip(self, fw: int, fh: int, hand_idx: int = 0) -> Optional[Tuple[int, int]]:
        return self.tip_px(self.MIDDLE_TIP, fw, fh, hand_idx=hand_idx)

    # ── Rendering ────────────────────────────────────────────

    def draw_landmarks(self, bgr: np.ndarray):
        for hand_lm in self._hand_landmarks:
            self._draw_utils.draw_landmarks(
                bgr,
                hand_lm,
                self._connections,
                self._draw_utils.DrawingSpec(color=(0, 230, 120), thickness=2, circle_radius=3),
                self._draw_utils.DrawingSpec(color=(0, 170, 90),  thickness=1),
            )


# ══════════════════════════════════════════════════════════════
#  GESTURE CONTROLLER
# ══════════════════════════════════════════════════════════════

class GestureController:
    """
    Maps finger-state dictionaries → stable Gesture enum values.
    Uses a majority-vote window to suppress per-frame noise.
    """

    def __init__(self):
        self._hist: deque = deque(maxlen=GESTURE_WIN)
        self.current: Gesture = Gesture.IDLE

    # ── Private ──────────────────────────────────────────────

    def _classify(
        self,
        states: Optional[Dict[str, bool]],
        cursor: Optional[Tuple[int, int]],
    ) -> Gesture:
        """Single-frame rule table (priority order)."""
        if states is None:
            return Gesture.IDLE

        t = states.get("thumb",  False)
        i = states.get("index",  False)
        m = states.get("middle", False)
        r = states.get("ring",   False)
        p = states.get("pinky",  False)

        if t and i and m and r and p:
            return Gesture.CLEAR    # Open palm
        if (not t) and i and m and r and p:
            return Gesture.PAN      # 4 fingers

        # More tolerant UI access: if index is up near toolbar, allow interaction.
        in_ui_zone = cursor is not None and cursor[1] < (TOOLBAR_H + UI_ACTIVE_MARGIN)
        aux_open = int(r) + int(p)
        if in_ui_zone and i and aux_open <= 1:
            return Gesture.UI_HOVER

        # Erase should trigger reliably when index+middle are up,
        # even if ring/pinky jitter for a frame.
        if i and m and aux_open <= 1:
            return Gesture.ERASE

        # Draw whenever index is up and other fingers are mostly down.
        if i and aux_open <= 1 and not m:
            return Gesture.DRAW

        return Gesture.IDLE

    # ── Public ───────────────────────────────────────────────

    def update(
        self,
        states: Optional[Dict[str, bool]],
        cursor: Optional[Tuple[int, int]],
    ) -> Gesture:
        """Call once per frame. Returns the temporally-stable gesture."""
        raw = self._classify(states, cursor)
        self._hist.append(raw)

        if len(self._hist) >= 3:
            counts: Dict[Gesture, int] = {}
            for g in self._hist:
                counts[g] = counts.get(g, 0) + 1
            self.current = max(counts, key=lambda g: counts[g])

        return self.current

    def reset(self):
        self._hist.clear()
        self.current = Gesture.IDLE


# ══════════════════════════════════════════════════════════════
#  UI BUTTON
# ══════════════════════════════════════════════════════════════

class UIButton:
    """
    A single interactive toolbar button.
    Activates after the cursor dwells on it for HOVER_TIME seconds.
    """

    def __init__(
        self,
        x: int, y: int, w: int, h: int,
        label: str,
        value=None,
        swatch: Optional[Tuple[int, int, int]] = None,
    ):
        self.x = x; self.y = y; self.w = w; self.h = h
        self.label    = label
        self.value    = value
        self.swatch   = swatch   # non-None → render as colour swatch
        self.selected = False

        self._hovered  = False
        self._hover_t0: Optional[float] = None
        self._fired    = False    # fire once per hover episode

    # ── Geometry ─────────────────────────────────────────────

    def hit(self, px: int, py: int) -> bool:
        margin = 12 if self._hovered else 7
        return (
            (self.x - margin) <= px <= (self.x + self.w + margin)
            and (self.y - margin) <= py <= (self.y + self.h + margin)
        )

    # ── Update ───────────────────────────────────────────────

    def update(self, cursor: Optional[Tuple[int, int]]) -> bool:
        """
        Feed the current cursor position each frame.
        Returns True **once** when the dwell-time activation fires.
        """
        over = (cursor is not None) and self.hit(*cursor)

        if over and not self._hovered:           # enter
            self._hover_t0 = time.time()
            self._fired    = False
        if not over:                             # leave
            self._hover_t0 = None
            self._fired    = False

        self._hovered = over

        if over and not self._fired and self._hover_t0 is not None:
            if time.time() - self._hover_t0 >= HOVER_TIME:
                self._fired = True
                return True        # ← activation event

        return False

    def progress(self) -> float:
        """0.0 – 1.0 hover fill progress."""
        if self._hovered and self._hover_t0 and not self._fired:
            return min(1.0, (time.time() - self._hover_t0) / HOVER_TIME)
        return 0.0

    # ── Rendering ────────────────────────────────────────────

    def draw(self, frame: np.ndarray):
        x, y, w, h = self.x, self.y, self.w, self.h
        prog = self.progress()

        if self.swatch is not None:
            # ── Colour swatch ─────────────────────────────
            cv2.rectangle(frame, (x, y), (x+w, y+h), self.swatch, -1)

            border_col = (255, 255, 255) if self.selected else \
                         (200, 200, 200) if self._hovered  else (60, 60, 60)
            border_t   = 3 if self.selected else (2 if self._hovered else 1)
            cv2.rectangle(frame,
                          (x - border_t, y - border_t),
                          (x + w + border_t, y + h + border_t),
                          border_col, border_t)

            # Dwell ring
            if prog > 0:
                cx, cy = x + w // 2, y + h // 2
                r = min(w, h) // 2
                cv2.ellipse(frame, (cx, cy), (r, r),
                            -90, 0, int(prog * 360), (255, 255, 255), 2)
        else:
            # ── Text button ───────────────────────────────
            bg = (50, 120, 200) if self.selected else \
                 (55, 55, 72)   if self._hovered  else (30, 30, 40)
            cv2.rectangle(frame, (x, y), (x+w, y+h), bg, -1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (75, 75, 75), 1)

            # Dwell fill bar at bottom
            if prog > 0:
                bw = int(w * prog)
                cv2.rectangle(frame, (x, y+h-3), (x+bw, y+h), (80, 200, 255), -1)

            # Label text
            font   = cv2.FONT_HERSHEY_SIMPLEX
            fscale = 0.37
            thick  = 1
            (tw, th), _ = cv2.getTextSize(self.label, font, fscale, thick)
            tx = x + (w - tw) // 2
            ty = y + (h + th) // 2 - 1
            cv2.putText(frame, self.label, (tx, ty), font, fscale, (220, 220, 220), thick)


# ══════════════════════════════════════════════════════════════
#  UI MANAGER
# ══════════════════════════════════════════════════════════════

class UIManager:
    """
    Owns and renders the top toolbar.
    Exposes `.tool`, `.color`, `.brush_size`, `.eraser_size` to the app.
    """

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height

        # ── Current selections (read by app loop) ────────────
        self.tool        = Tool.PEN
        self.gesture_mode = GestureInputMode.ONE_HAND
        self.color       = PALETTE["White"]
        self.color_name  = "White"
        self.brush_size  = BRUSH_SIZES[1]       # default 6
        self.eraser_size = ERASER_SIZES[1]      # default 35

        # ── Notification banner ──────────────────────────────
        self._notif      = ""
        self._notif_end  = 0.0

        # ── Build toolbar ────────────────────────────────────
        self._buttons:      List[UIButton] = []
        self._color_btns:   List[UIButton] = []
        self._brush_btns:   List[UIButton] = []
        self._eraser_btns:  List[UIButton] = []
        self._btn_pen:      UIButton
        self._btn_eraser:   UIButton
        self._btn_mode_1h:  UIButton
        self._btn_mode_2h:  UIButton
        self._build()

    # ── Layout construction ──────────────────────────────────

    def _build(self):
        BH = 46                            # button height
        BY = (TOOLBAR_H - BH) // 2        # vertical centre
        x  = 8

        def add(btn: UIButton) -> UIButton:
            self._buttons.append(btn)
            return btn

        # Tool buttons
        self._btn_pen    = add(UIButton(x, BY, 60, BH, "Pen",    value=Tool.PEN));    x += 68
        self._btn_eraser = add(UIButton(x, BY, 70, BH, "Eraser", value=Tool.ERASER)); x += 80

        x += 4
        self._btn_mode_1h = add(UIButton(x, BY, 44, BH, "1H", value=("mode", "1h"))); x += 50
        self._btn_mode_2h = add(UIButton(x, BY, 44, BH, "2H", value=("mode", "2h"))); x += 52

        x += 6  # section gap

        # Colour swatches (38×38 each)
        SW = 38; SY = (TOOLBAR_H - SW) // 2
        for cname, bgr in PALETTE.items():
            btn = UIButton(x, SY, SW, SW, cname, value=bgr, swatch=bgr)
            self._color_btns.append(btn)
            self._buttons.append(btn)
            x += SW + 5

        x += 6

        # Brush size buttons
        for sz in BRUSH_SIZES:
            btn = UIButton(x, BY + 4, 30, BH - 8, str(sz), value=("brush", sz))
            self._brush_btns.append(btn)
            self._buttons.append(btn)
            x += 36

        x += 6

        # Eraser size buttons
        for sz, lbl in zip(ERASER_SIZES, ["S", "M", "L", "XL"]):
            btn = UIButton(x, BY + 4, 26, BH - 8, lbl, value=("eraser", sz))
            self._eraser_btns.append(btn)
            self._buttons.append(btn)
            x += 32

        x += 6

        # Action buttons
        add(UIButton(x, BY, 62, BH, "Clear", value="clear")); x += 70
        add(UIButton(x, BY, 62, BH, "Save",  value="save"));  x += 70
        add(UIButton(x, BY, 62, BH, "Undo",  value="undo"));  x += 70
        add(UIButton(x, BY, 58, BH, "Z-",    value="zoom_out")); x += 66
        add(UIButton(x, BY, 58, BH, "Z+",    value="zoom_in"))

    # ── Tick ─────────────────────────────────────────────────

    def update(self, cursor: Optional[Tuple[int, int]]) -> List[str]:
        """
        Call each frame.  `cursor` should be None when finger is outside
        the toolbar zone so buttons reset their hover state.
        Returns a list of action strings fired this frame.
        """
        actions: List[str] = []
        for btn in self._buttons:
            if btn.update(cursor):
                act = self._handle(btn)
                if act:
                    actions.append(act)
        self._sync_selection()
        return actions

    def click(self, cursor: Tuple[int, int]) -> Optional[str]:
        """
        Immediate mouse/trackpad click activation for toolbar items.
        """
        for btn in self._buttons:
            if btn.hit(*cursor):
                act = self._handle(btn)
                self._sync_selection()
                return act
        return None

    def _handle(self, btn: UIButton) -> Optional[str]:
        v = btn.value

        if v == Tool.PEN:
            self.tool = Tool.PEN
            self.notify("Pen selected")

        elif v == Tool.ERASER:
            self.tool = Tool.ERASER
            self.notify("Eraser selected")

        elif isinstance(v, tuple) and v[0] == "brush":
            self.brush_size = v[1]
            self.notify(f"Brush size: {v[1]}")

        elif isinstance(v, tuple) and v[0] == "eraser":
            self.eraser_size = v[1]
            self.notify(f"Eraser size: {v[1]}")

        elif isinstance(v, tuple) and len(v) == 2 and v[0] == "mode":
            mode = GestureInputMode.ONE_HAND if v[1] == "1h" else GestureInputMode.TWO_HAND
            if mode != self.gesture_mode:
                self.gesture_mode = mode
                lbl = "One-hand mode" if mode == GestureInputMode.ONE_HAND else "Two-hand mode"
                self.notify(lbl)

        elif isinstance(v, tuple) and len(v) == 3:
            # colour tuple (BGR)
            self.color      = v
            self.color_name = btn.label
            self.notify(f"Colour: {btn.label}")

        elif v == "clear":  return "clear"
        elif v == "save":   return "save"
        elif v == "undo":   return "undo"
        elif v == "zoom_in":  return "zoom_in"
        elif v == "zoom_out": return "zoom_out"

        return None

    def _sync_selection(self):
        self._btn_pen.selected    = (self.tool == Tool.PEN)
        self._btn_eraser.selected = (self.tool == Tool.ERASER)
        self._btn_mode_1h.selected = (self.gesture_mode == GestureInputMode.ONE_HAND)
        self._btn_mode_2h.selected = (self.gesture_mode == GestureInputMode.TWO_HAND)
        for b in self._color_btns:
            b.selected = (b.value == self.color)
        for b in self._brush_btns:
            b.selected = (b.value == ("brush", self.brush_size))
        for b in self._eraser_btns:
            b.selected = (b.value == ("eraser", self.eraser_size))

    # ── Notification ─────────────────────────────────────────

    def notify(self, text: str, duration: float = 2.0):
        self._notif     = text
        self._notif_end = time.time() + duration

    # ── Rendering ────────────────────────────────────────────

    def draw(self, frame: np.ndarray):
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, TOOLBAR_H), (16, 16, 24), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.line(frame, (0, TOOLBAR_H), (self.width, TOOLBAR_H), (80, 80, 80), 1)

        # Section micro-labels
        _lbl = lambda txt, px: cv2.putText(
            frame, txt, (px, TOOLBAR_H - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (90, 90, 100), 1
        )
        _lbl("TOOLS",   8)
        _lbl("MODE",    160)
        _lbl("COLOURS", 270)
        _lbl("BRUSH",   620)
        _lbl("ERASER",  805)
        _lbl("ACTIONS", 940)

        for btn in self._buttons:
            btn.draw(frame)

        # Current-colour dot (top-right corner)
        cr = (self.width - 18, 18)
        cv2.circle(frame, cr, 11, self.color, -1)
        cv2.circle(frame, cr, 11, (110, 110, 110), 1)

        # Notification banner
        if self._notif and time.time() < self._notif_end:
            self._draw_notif(frame)

    def _draw_notif(self, frame: np.ndarray):
        msg  = self._notif
        font = cv2.FONT_HERSHEY_SIMPLEX
        fsc  = 0.62; th = 2
        (tw, ht), _ = cv2.getTextSize(msg, font, fsc, th)
        pad  = 12
        nx   = (self.width - tw) // 2
        ny   = TOOLBAR_H + 38
        cv2.rectangle(frame, (nx-pad, ny-ht-pad//2), (nx+tw+pad, ny+pad//2), (10,10,18), -1)
        cv2.rectangle(frame, (nx-pad, ny-ht-pad//2), (nx+tw+pad, ny+pad//2), (70,180,255), 1)
        cv2.putText(frame, msg, (nx, ny), font, fsc, (70, 200, 255), th)


# ══════════════════════════════════════════════════════════════
#  DRAWING CANVAS
# ══════════════════════════════════════════════════════════════

class DrawingCanvas:
    """
    Manages the drawing surface as a dedicated NumPy layer.

    Features
    ────────
    • Smooth stroke interpolation (no gaps between samples)
    • Speed-aware brush size (smaller when moving slowly → precision)
    • Weighted moving-average for jitter reduction
    • Per-stroke undo history (up to MAX_UNDO snapshots)
    • Canvas panning via pan_x / pan_y offset
    • PNG export with timestamp filename
    """

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        self._cvs: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        self._hist: List[np.ndarray] = []

        # Stroke state
        self._prev_pt: Optional[Tuple[int, int]] = None
        self._sx: deque = deque(maxlen=STROKE_SMOOTH)
        self._sy: deque = deque(maxlen=STROKE_SMOOTH)

        # Pan offset (applied at composite time)
        self.pan_x = 0
        self.pan_y = 0
        self.zoom = 1.0
        self._min_zoom = 0.6
        self._max_zoom = 3.0

    # ── History / undo ───────────────────────────────────────

    def _push(self):
        if len(self._hist) >= MAX_UNDO:
            self._hist.pop(0)
        self._hist.append(self._cvs.copy())

    def undo(self):
        if self._hist:
            self._cvs = self._hist.pop()
            self._reset_stroke()

    def clear(self):
        self._push()
        self._cvs[:] = 0
        self._reset_stroke()

    # ── Stroke helpers ───────────────────────────────────────

    def _reset_stroke(self):
        self._prev_pt = None
        self._sx.clear()
        self._sy.clear()

    def _smooth(self, x: int, y: int) -> Tuple[int, int]:
        """Weighted moving average (recent samples weigh more)."""
        self._sx.append(x); self._sy.append(y)
        n  = len(self._sx)
        w  = np.arange(1, n + 1, dtype=float); w /= w.sum()
        return int(np.dot(w, list(self._sx))), int(np.dot(w, list(self._sy)))

    def _canvas_coord(self, sx: int, sy: int) -> Tuple[int, int]:
        """Map screen position → canvas position (accounts for pan + zoom)."""
        return (
            int(round((sx - self.pan_x) / self.zoom)),
            int(round((sy - self.pan_y) / self.zoom)),
        )

    def _precision_size(self, cx: int, cy: int, base: int) -> int:
        """
        Slow movement → slightly thinner stroke (higher precision).
        Fast movement → full base size (smooth appearance).
        """
        if self._prev_pt is None:
            return base
        dist   = math.hypot(cx - self._prev_pt[0], cy - self._prev_pt[1])
        factor = max(0.55, min(1.0, dist / 18.0))
        return max(1, int(base * factor))

    def _stamp(self, canvas_pt: Tuple[int,int], size: int, color: Tuple[int,int,int]):
        px, py = canvas_pt
        if 0 <= px < self.width and 0 <= py < self.height:
            cv2.circle(self._cvs, (px, py), size, color, -1)

    # ── Public drawing ops ───────────────────────────────────

    def begin_stroke(self):
        """Call when a new stroke starts; saves an undo snapshot."""
        self._push()
        self._reset_stroke()

    def end_stroke(self):
        """Call when finger lifts / gesture changes."""
        self._reset_stroke()

    def draw_at(self, sx: int, sy: int,
                color: Tuple[int, int, int], base_size: int):
        """
        Draw a smooth continuous stroke at screen position (sx, sy).
        Uses sub-pixel interpolation between consecutive samples.
        """
        cx, cy = self._canvas_coord(sx, sy)
        cx, cy = self._smooth(cx, cy)
        size   = self._precision_size(cx, cy, base_size)

        if self._prev_pt is not None:
            px, py = self._prev_pt
            dist   = math.hypot(cx - px, cy - py)
            steps  = max(1, int(dist))          # 1 stamp per pixel of travel
            for i in range(steps + 1):
                t  = i / steps
                ix = int(px + t * (cx - px))
                iy = int(py + t * (cy - py))
                if 0 <= ix < self.width and 0 <= iy < self.height:
                    cv2.circle(self._cvs, (ix, iy), size, color, -1)
        else:
            self._stamp((cx, cy), size, color)

        self._prev_pt = (cx, cy)

    def erase_at(self, sx: int, sy: int, eraser_size: int):
        """Erase a circular region; interpolates between samples."""
        cx, cy = self._canvas_coord(sx, sy)
        cx, cy = self._smooth(cx, cy)

        if self._prev_pt is not None:
            px, py = self._prev_pt
            dist   = math.hypot(cx - px, cy - py)
            steps  = max(1, int(dist))
            for i in range(steps + 1):
                t  = i / steps
                ix = int(px + t * (cx - px))
                iy = int(py + t * (cy - py))
                if 0 <= ix < self.width and 0 <= iy < self.height:
                    cv2.circle(self._cvs, (ix, iy), eraser_size, (0, 0, 0), -1)
        else:
            self._stamp((cx, cy), eraser_size, (0, 0, 0))

        self._prev_pt = (cx, cy)

    def pan(self, dx: int, dy: int):
        self.pan_x += dx
        self.pan_y += dy
        self._clamp_view()

    def zoom_by(self, delta: float, anchor: Optional[Tuple[int, int]] = None) -> bool:
        old_zoom = self.zoom
        new_zoom = float(np.clip(old_zoom + delta, self._min_zoom, self._max_zoom))
        if abs(new_zoom - old_zoom) < 1e-6:
            return False

        if anchor is None:
            anchor = (self.width // 2, self.height // 2)
        ax, ay = anchor

        # Keep anchor point stable while zooming.
        self.pan_x = int(round(ax - ((ax - self.pan_x) / old_zoom) * new_zoom))
        self.pan_y = int(round(ay - ((ay - self.pan_y) / old_zoom) * new_zoom))
        self.zoom = new_zoom
        self._clamp_view()
        self._reset_stroke()
        return True

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._reset_stroke()

    # ── Compositing ──────────────────────────────────────────

    def composite(self, camera_frame: np.ndarray) -> np.ndarray:
        """
        Overlay drawing onto camera frame.
        Canvas pixels with content (non-black) replace camera pixels.
        """
        result = camera_frame.copy()

        z_w, z_h = self._zoomed_size()
        if abs(self.zoom - 1.0) < 1e-6:
            scaled = self._cvs
        else:
            interp = cv2.INTER_LINEAR if self.zoom >= 1.0 else cv2.INTER_AREA
            scaled = cv2.resize(self._cvs, (z_w, z_h), interpolation=interp)

        disp = np.zeros_like(self._cvs)
        sx1 = max(0, -self.pan_x)
        sy1 = max(0, -self.pan_y)
        dx1 = max(0, self.pan_x)
        dy1 = max(0, self.pan_y)
        cw = min(self.width - dx1, scaled.shape[1] - sx1)
        ch = min(self.height - dy1, scaled.shape[0] - sy1)
        if cw > 0 and ch > 0:
            disp[dy1:dy1+ch, dx1:dx1+cw] = scaled[sy1:sy1+ch, sx1:sx1+cw]

        # Boolean mask: anywhere the canvas has content
        mask = np.any(disp > 12, axis=2)    # slight threshold avoids jpeg noise
        result[mask] = disp[mask]
        return result

    def _zoomed_size(self) -> Tuple[int, int]:
        return (
            max(1, int(round(self.width * self.zoom))),
            max(1, int(round(self.height * self.zoom))),
        )

    def _clamp_view(self):
        z_w, z_h = self._zoomed_size()

        if z_w >= self.width:
            min_x, max_x = self.width - z_w, 0
        else:
            min_x, max_x = 0, self.width - z_w

        if z_h >= self.height:
            min_y, max_y = self.height - z_h, 0
        else:
            min_y, max_y = 0, self.height - z_h

        self.pan_x = int(np.clip(self.pan_x, min_x, max_x))
        self.pan_y = int(np.clip(self.pan_y, min_y, max_y))

    # ── Export ───────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        if path is None:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"air_writing_{ts}.png"
        cv2.imwrite(path, self._cvs)
        return path


# ══════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════

class AirWritingPro:
    """
    Top-level orchestrator.
    Owns the capture loop, all subsystems, keyboard shortcuts, and HUD.
    """

    def __init__(self):
        # ── Camera ──────────────────────────────────────────
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Cannot open camera.\n"
                "On macOS: System Preferences → Security → Camera → allow Terminal / IDE."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self._cap.set(cv2.CAP_PROP_FPS,            30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,      1)   # minimise latency

        ret, probe = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read initial frame from camera.")
        self._fh, self._fw = probe.shape[:2]

        # ── Subsystems ──────────────────────────────────────
        self._tracker  = HandTracker()
        self._gesture  = GestureController()
        self._canvas   = DrawingCanvas(self._fw, self._fh)
        self._ui       = UIManager(self._fw, self._fh)

        # ── Cursor smoothing ─────────────────────────────────
        self._cursor_ema: Optional[Tuple[float, float]] = None
        self._pending_mouse_actions: List[str] = []

        # ── State flags ──────────────────────────────────────
        self._stroke_active = False
        self._erase_active  = False
        self._pan_prev: Optional[Tuple[int, int]] = None

        # Clear-gesture debounce
        self._last_clear = 0.0
        self._last_dual_undo = 0.0
        self._dual_zoom_prev_dist: Optional[float] = None
        self._dual_pan_prev_center: Optional[Tuple[int, int]] = None

        # ── FPS ──────────────────────────────────────────────
        self._fps_buf: deque = deque(maxlen=30)
        self._last_t = time.time()

        self._print_welcome()

    # ── Welcome ──────────────────────────────────────────────

    @staticmethod
    def _print_welcome():
        print()
        print("╔════════════════════════════════════════════╗")
        print("║       Air Writing Pro  —  Ready            ║")
        print("╠════════════════════════════════════════════╣")
        print("║  Gestures:                                 ║")
        print("║    Index only       → Draw                 ║")
        print("║    Index + Middle   → Erase                ║")
        print("║    Open palm        → Clear (hold 1 s)     ║")
        print("║    4 fingers        → Pan                  ║")
        print("║    Finger on HUD    → Select tool / colour ║")
        print("║    Toolbar 1H / 2H  → Switch gesture mode  ║")
        print("║    2H Index pinch   → Zoom in/out          ║")
        print("║    2H 4-fingers     → Pan canvas           ║")
        print("║    2H open palms    → Clear canvas         ║")
        print("║    2H fists         → Undo                 ║")
        print("║    Mouse/Trackpad   → Click toolbar buttons║")
        print("╠════════════════════════════════════════════╣")
        print("║  Keys: q=quit c=clear u=undo s=save +/-=zoom║")
        print("╚════════════════════════════════════════════╝")
        print()

    # ── Helpers ──────────────────────────────────────────────

    def _smooth_cursor(self, raw: Tuple[int, int]) -> Tuple[int, int]:
        if self._cursor_ema is None:
            self._cursor_ema = (float(raw[0]), float(raw[1]))
            return raw

        sx, sy = self._cursor_ema
        dx = raw[0] - sx
        dy = raw[1] - sy
        dist = math.hypot(dx, dy)
        gain = min(1.0, dist / CURSOR_FAST_PX)
        alpha = CURSOR_ALPHA_SLOW + gain * (CURSOR_ALPHA_FAST - CURSOR_ALPHA_SLOW)

        sx = (1.0 - alpha) * sx + alpha * raw[0]
        sy = (1.0 - alpha) * sy + alpha * raw[1]
        self._cursor_ema = (sx, sy)
        return int(round(sx)), int(round(sy))

    def _tick_fps(self) -> float:
        now = time.time()
        dt  = now - self._last_t
        self._last_t = now
        if dt > 0:
            self._fps_buf.append(1.0 / dt)
        return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0

    def _try_clear(self):
        now = time.time()
        if now - self._last_clear > CLEAR_COOLDOWN:
            self._canvas.clear()
            self._ui.notify("Canvas cleared!", 1.5)
            self._last_clear = now

    def _handle_ui_action(self, act: str):
        if act == "clear":
            self._canvas.clear()
            self._ui.notify("Canvas cleared!")
        elif act == "save":
            fn = self._canvas.save()
            self._ui.notify(f"Saved: {fn}")
            print(f"[Save] {fn}")
        elif act == "undo":
            self._canvas.undo()
            self._ui.notify("Undo!")
        elif act == "zoom_in":
            if self._canvas.zoom_by(+0.15):
                self._ui.notify(f"Zoom: {self._canvas.zoom:.2f}x")
        elif act == "zoom_out":
            if self._canvas.zoom_by(-0.15):
                self._ui.notify(f"Zoom: {self._canvas.zoom:.2f}x")

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        act = self._ui.click((x, y))
        if act:
            self._pending_mouse_actions.append(act)

    def _pick_primary_hand(self, hands: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if not hands:
            return None
        # Prefer reported "Right" hand, else the right-most hand in frame.
        for hand in hands:
            if str(hand.get("label", "")).lower() == "right":
                return hand
        return max(hands, key=lambda h: int(h["center"][0]))

    @staticmethod
    def _is_open_palm(states: Dict[str, bool]) -> bool:
        return all(states.get(k, False) for k in ("thumb", "index", "middle", "ring", "pinky"))

    @staticmethod
    def _is_four_fingers(states: Dict[str, bool]) -> bool:
        return (not states.get("thumb", False)) and all(
            states.get(k, False) for k in ("index", "middle", "ring", "pinky")
        )

    @staticmethod
    def _is_fist(states: Dict[str, bool]) -> bool:
        return sum(int(states.get(k, False)) for k in ("thumb", "index", "middle", "ring", "pinky")) <= 1

    @staticmethod
    def _is_index_zoom_pose(states: Dict[str, bool]) -> bool:
        return (
            states.get("index", False)
            and not states.get("middle", False)
            and (int(states.get("ring", False)) + int(states.get("pinky", False)) <= 1)
        )

    def _reset_dual_trackers(self):
        self._dual_zoom_prev_dist = None
        self._dual_pan_prev_center = None

    def _handle_dual_hand_gestures(self, hands: List[Dict[str, object]]) -> Optional[str]:
        if len(hands) < 2:
            self._reset_dual_trackers()
            return None

        # Use two most separated hands in X for stability.
        hands2 = sorted(hands, key=lambda h: int(h["center"][0]))
        h1, h2 = hands2[0], hands2[-1]
        s1 = h1["states"]
        s2 = h2["states"]
        p1 = h1["index_tip"]
        p2 = h2["index_tip"]

        if not isinstance(s1, dict) or not isinstance(s2, dict):
            self._reset_dual_trackers()
            return None

        # 1) Both palms open -> clear (debounced).
        if self._is_open_palm(s1) and self._is_open_palm(s2):
            self._try_clear()
            self._reset_dual_trackers()
            return "2H CLEAR"

        # 2) Both fists -> undo (cooldown).
        now = time.time()
        if self._is_fist(s1) and self._is_fist(s2):
            if now - self._last_dual_undo > DUAL_UNDO_COOLDOWN:
                self._canvas.undo()
                self._ui.notify("Undo (2H)")
                self._last_dual_undo = now
            self._reset_dual_trackers()
            return "2H UNDO"

        # 3) Two-hand pinch (both index pose) -> zoom.
        if self._is_index_zoom_pose(s1) and self._is_index_zoom_pose(s2) and p1 and p2:
            dist = math.hypot(float(p1[0] - p2[0]), float(p1[1] - p2[1]))
            if self._dual_zoom_prev_dist is not None:
                dd = dist - self._dual_zoom_prev_dist
                if abs(dd) >= 1.0:
                    step = float(np.clip(dd * 0.004, -0.08, 0.08))
                    anchor = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    self._canvas.zoom_by(step, anchor=anchor)
            self._dual_zoom_prev_dist = dist
            self._dual_pan_prev_center = None
            return "2H ZOOM"
        self._dual_zoom_prev_dist = None

        # 4) Two hands 4-finger pose -> pan.
        if self._is_four_fingers(s1) and self._is_four_fingers(s2):
            c1 = h1["center"]
            c2 = h2["center"]
            center = ((int(c1[0]) + int(c2[0])) // 2, (int(c1[1]) + int(c2[1])) // 2)
            if self._dual_pan_prev_center is not None:
                dx = center[0] - self._dual_pan_prev_center[0]
                dy = center[1] - self._dual_pan_prev_center[1]
                self._canvas.pan(dx, dy)
            self._dual_pan_prev_center = center
            return "2H PAN"
        self._dual_pan_prev_center = None

        return None

    # ── HUD rendering ────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        gesture: Gesture,
        cursor: Optional[Tuple[int, int]],
        dual_status: Optional[str] = None,
    ):
        fh, fw = frame.shape[:2]
        font   = cv2.FONT_HERSHEY_SIMPLEX

        # FPS — bottom-left
        fps_col = (0, 240, 80) if fps >= 24 else \
                  (0, 210, 255) if fps >= 15 else (0, 80, 255)
        cv2.putText(frame, f"FPS {fps:.0f}",
                    (10, fh - 12), font, 0.55, fps_col, 2)

        # Gesture label — bottom-centre
        gmap = {
            Gesture.IDLE:     ("IDLE",     (110, 110, 110)),
            Gesture.DRAW:     ("DRAWING",  (60,  245, 60)),
            Gesture.ERASE:    ("ERASING",  (60,  100, 255)),
            Gesture.CLEAR:    ("CLEAR!",   (40,  40,  255)),
            Gesture.PAN:      ("PANNING",  (255, 195, 40)),
            Gesture.UI_HOVER: ("UI MODE",  (255, 120, 40)),
        }
        glbl, gcol = gmap.get(gesture, ("?", (180, 180, 180)))
        tw = cv2.getTextSize(glbl, font, 0.60, 2)[0][0]
        cv2.putText(frame, glbl, ((fw - tw) // 2, fh - 12),
                    font, 0.60, gcol, 2)

        # Tool + colour + mode — bottom-right
        mode_lbl = "2H" if self._ui.gesture_mode == GestureInputMode.TWO_HAND else "1H"
        info = f"{mode_lbl}  {self._ui.tool.value.upper()}  {self._ui.color_name}"
        tw = cv2.getTextSize(info, font, 0.48, 1)[0][0]
        cv2.putText(frame, info, (fw - tw - 10, fh - 12),
                    font, 0.48, (170, 170, 170), 1)

        if dual_status:
            cv2.putText(frame, dual_status, (10, fh - 56), font, 0.45, (120, 210, 255), 1)

        # Pan / zoom indicator
        if self._canvas.pan_x != 0 or self._canvas.pan_y != 0 or abs(self._canvas.zoom - 1.0) > 1e-3:
            pan_txt = (
                f"Pan  dx={self._canvas.pan_x:+d}  dy={self._canvas.pan_y:+d}    "
                f"Zoom {self._canvas.zoom:.2f}x"
            )
            cv2.putText(frame, pan_txt, (10, fh - 34), font, 0.38, (160, 160, 90), 1)

        # ── Cursor visualisation ─────────────────────────────
        if cursor:
            cx, cy = cursor
            if gesture == Gesture.DRAW:
                sz  = self._ui.brush_size
                col = self._ui.color
                cv2.circle(frame, (cx, cy), sz + 2, col, -1)
                cv2.circle(frame, (cx, cy), sz + 4, (255, 255, 255), 1)

            elif gesture == Gesture.ERASE:
                sz = self._ui.eraser_size
                cv2.circle(frame, (cx, cy), sz, (190, 190, 190), 2)
                cv2.line(frame, (cx-5, cy),   (cx+5, cy),   (190, 190, 190), 1)
                cv2.line(frame, (cx, cy-5),   (cx, cy+5),   (190, 190, 190), 1)

            elif gesture == Gesture.UI_HOVER:
                cv2.circle(frame, (cx, cy), 10, (255, 195, 40), -1)
                cv2.circle(frame, (cx, cy), 13, (255, 255, 255), 2)

            elif gesture == Gesture.PAN:
                cv2.drawMarker(frame, (cx, cy), (255, 195, 40),
                               cv2.MARKER_CROSS, 22, 2)

            else:
                cv2.circle(frame, (cx, cy), 5, (190, 190, 190), -1)

        # ── Clear warning overlay ─────────────────────────────
        if gesture == Gesture.CLEAR:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (fw, fh), (0, 0, 50), -1)
            cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)
            msg = "OPEN PALM  =  CLEAR CANVAS"
            tw  = cv2.getTextSize(msg, font, 0.95, 2)[0][0]
            cv2.putText(frame, msg, ((fw - tw) // 2, fh // 2),
                        font, 0.95, (50, 50, 255), 2)

    # ── Main event loop ──────────────────────────────────────

    def run(self):
        cv2.namedWindow("Air Writing Pro", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Air Writing Pro", self._fw, self._fh)
        cv2.setMouseCallback("Air Writing Pro", self._on_mouse)

        while True:
            # ── Capture + mirror ─────────────────────────────
            ret, frame = self._cap.read()
            if not ret:
                print("[WARN] Dropped frame, retrying …")
                time.sleep(0.03)
                continue
            frame = cv2.flip(frame, 1)

            # ── Hand tracking ────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._tracker.process(rgb)

            hands = self._tracker.hand_infos(self._fw, self._fh)
            primary = self._pick_primary_hand(hands)
            fstates = primary["states"] if primary else None
            raw_pos = primary["index_tip"] if primary else None
            cursor  = self._smooth_cursor(raw_pos) if raw_pos else None

            # ── Gesture ──────────────────────────────────────
            gesture = self._gesture.update(fstates, cursor)
            dual_status: Optional[str] = None
            if self._ui.gesture_mode == GestureInputMode.TWO_HAND:
                dual_status = self._handle_dual_hand_gestures(hands)
            else:
                self._reset_dual_trackers()

            # ── UI ───────────────────────────────────────────
            # Pass cursor when it is in/near toolbar zone for more stable interaction.
            ui_cur   = cursor if (cursor and cursor[1] < (TOOLBAR_H + UI_ACTIVE_MARGIN)) else None
            ui_acts  = self._ui.update(ui_cur)
            if self._pending_mouse_actions:
                ui_acts.extend(self._pending_mouse_actions)
                self._pending_mouse_actions.clear()

            for act in ui_acts:
                self._handle_ui_action(act)

            # ── Canvas ops ───────────────────────────────────
            in_zone = bool(cursor and cursor[1] >= (TOOLBAR_H + UI_ACTIVE_MARGIN))
            dual_active = dual_status is not None

            if in_zone and cursor is not None and not dual_active:
                eff_tool  = self._ui.tool
                is_draw   = (gesture == Gesture.DRAW   and eff_tool == Tool.PEN)
                is_erase  = (gesture == Gesture.ERASE) or \
                            (gesture == Gesture.DRAW   and eff_tool == Tool.ERASER)

                if is_draw:
                    if not self._stroke_active:
                        self._canvas.begin_stroke()
                        self._stroke_active = True
                    self._erase_active = False
                    self._canvas.draw_at(cursor[0], cursor[1],
                                         self._ui.color, self._ui.brush_size)

                elif is_erase:
                    if not self._erase_active:
                        self._canvas.begin_stroke()  # save state before erasing
                        self._erase_active = True
                    self._stroke_active = False
                    self._canvas.erase_at(cursor[0], cursor[1],
                                          self._ui.eraser_size)
                else:
                    if self._stroke_active or self._erase_active:
                        self._canvas.end_stroke()
                    self._stroke_active = False
                    self._erase_active  = False

                # Pan ─────────────────────────────────────────
                if gesture == Gesture.PAN:
                    if self._pan_prev is not None:
                        dx = cursor[0] - self._pan_prev[0]
                        dy = cursor[1] - self._pan_prev[1]
                        self._canvas.pan(dx, dy)
                    self._pan_prev = cursor
                else:
                    self._pan_prev = None

                # Clear (debounced) ───────────────────────────
                if gesture == Gesture.CLEAR:
                    self._try_clear()

            else:
                # Finger outside draw zone
                if self._stroke_active or self._erase_active:
                    self._canvas.end_stroke()
                self._stroke_active = False
                self._erase_active  = False
                self._pan_prev      = None

            # ── Render ───────────────────────────────────────
            output = self._canvas.composite(frame)
            self._tracker.draw_landmarks(output)
            self._ui.draw(output)
            self._draw_hud(output, self._tick_fps(), gesture, cursor, dual_status=dual_status)

            cv2.imshow("Air Writing Pro", output)

            # ── Keyboard shortcuts ───────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'):
                print("Quitting …")
                break
            elif key == ord('c'):
                self._canvas.clear();         self._ui.notify("Canvas cleared!")
            elif key == ord('u'):
                self._canvas.undo();          self._ui.notify("Undo!")
            elif key == ord('s'):
                fn = self._canvas.save()
                self._ui.notify(f"Saved: {fn}")
                print(f"[Save] {fn}")
            elif key in (ord('+'), ord('=')):
                if self._canvas.zoom_by(+0.15):
                    self._ui.notify(f"Zoom: {self._canvas.zoom:.2f}x")
            elif key in (ord('-'), ord('_')):
                if self._canvas.zoom_by(-0.15):
                    self._ui.notify(f"Zoom: {self._canvas.zoom:.2f}x")
            elif key == ord('0'):
                self._canvas.reset_view()
                self._ui.notify("View reset")

        # ── Cleanup ──────────────────────────────────────────
        self._cap.release()
        cv2.destroyAllWindows()
        print("Air Writing Pro closed.")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        AirWritingPro().run()
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}\n")
    except KeyboardInterrupt:
        print("\nInterrupted — bye!")
