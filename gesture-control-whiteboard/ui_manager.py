"""
ui_manager.py
─────────────
Toolbar UI: buttons, colour swatches, hover-dwell activation,
notification banners, and current-tool state management.
"""

import cv2
import time
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ── Toolbar geometry ─────────────────────────────────────────
TOOLBAR_H   = 80        # pixel height of the top bar
HOVER_TIME  = 0.55      # seconds a finger must dwell to activate a button
UI_ACTIVE_MARGIN = 24   # allow slight extra height for easier toolbar interactions

# ── Colour palette (BGR) ─────────────────────────────────────
PALETTE: Dict[str, Tuple[int, int, int]] = {
    "White":   (255, 255, 255),
    "Red":     (40,  40,  220),
    "Blue":    (210, 80,  20),
    "Green":   (40,  200, 40),
    "Yellow":  (0,   215, 215),
    "Cyan":    (210, 210, 0),
    "Magenta": (195, 40,  195),
    "Orange":  (0,   140, 255),
}

# ── Size options ─────────────────────────────────────────────
BRUSH_SIZES  = [3, 6, 10, 16, 24]
ERASER_SIZES = [20, 35, 55, 75]


class Tool(Enum):
    PEN    = "pen"
    ERASER = "eraser"


class GestureInputMode(Enum):
    ONE_HAND = "1h"
    TWO_HAND = "2h"


# ══════════════════════════════════════════════════════════════
#  UI BUTTON
# ══════════════════════════════════════════════════════════════

class UIButton:
    """
    A single interactive element.
    Fires once after the cursor dwells on it for `HOVER_TIME` seconds.
    Renders either as a labelled text button or a coloured swatch.
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
        self.swatch   = swatch      # non-None → render as colour square
        self.selected = False

        self._hovered  = False
        self._hover_t0: Optional[float] = None
        self._fired    = False

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
        Feed the cursor position.
        Returns True exactly once per hover episode when dwell completes.
        """
        over = (cursor is not None) and self.hit(*cursor)

        if over and not self._hovered:      # enter
            self._hover_t0 = time.time()
            self._fired    = False
        if not over:                        # leave
            self._hover_t0 = None
            self._fired    = False

        self._hovered = over

        if over and not self._fired and self._hover_t0 is not None:
            if time.time() - self._hover_t0 >= HOVER_TIME:
                self._fired = True
                return True

        return False

    def progress(self) -> float:
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

            bc = (255,255,255) if self.selected else \
                 (200,200,200) if self._hovered  else (55, 55, 55)
            bt = 3 if self.selected else 2 if self._hovered else 1
            cv2.rectangle(frame, (x-bt, y-bt), (x+w+bt, y+h+bt), bc, bt)

            if prog > 0:
                cx_c = x + w // 2;  cy_c = y + h // 2
                r = min(w, h) // 2
                cv2.ellipse(frame, (cx_c, cy_c), (r, r),
                            -90, 0, int(prog * 360), (255, 255, 255), 2)
        else:
            # ── Text button ───────────────────────────────
            bg = (50, 120, 200) if self.selected else \
                 (55, 55, 72)   if self._hovered  else (28, 28, 38)
            cv2.rectangle(frame, (x, y), (x+w, y+h), bg, -1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (72, 72, 72), 1)

            if prog > 0:
                bw = int(w * prog)
                cv2.rectangle(frame, (x, y+h-3), (x+bw, y+h), (75, 195, 255), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX; fs = 0.37; tk = 1
            (tw, th), _ = cv2.getTextSize(self.label, font, fs, tk)
            cv2.putText(frame, self.label,
                        (x + (w-tw)//2, y + (h+th)//2 - 1),
                        font, fs, (220, 220, 220), tk)


# ══════════════════════════════════════════════════════════════
#  UI MANAGER
# ══════════════════════════════════════════════════════════════

class UIManager:
    """
    Owns the full toolbar: tool buttons, colour palette, size selectors,
    and action buttons (Clear / Save / Undo).

    Public state (read by the app loop)
    ────────────────────────────────────
        .tool          Tool.PEN | Tool.ERASER
        .color         (B, G, R) tuple
        .color_name    human-readable colour name
        .brush_size    int in BRUSH_SIZES
        .eraser_size   int in ERASER_SIZES

    Action protocol
    ───────────────
        actions = ui.update(cursor)   # list of str: "clear","save","undo"
        ui.draw(frame)
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.fw = frame_width
        self.fh = frame_height

        # ── Current tool state ───────────────────────────────
        self.tool        = Tool.PEN
        self.gesture_mode = GestureInputMode.ONE_HAND
        self.color       = PALETTE["White"]
        self.color_name  = "White"
        self.brush_size  = BRUSH_SIZES[1]
        self.eraser_size = ERASER_SIZES[1]

        # ── Notification ─────────────────────────────────────
        self._notif     = ""
        self._notif_end = 0.0

        # ── Buttons ──────────────────────────────────────────
        self._buttons:     List[UIButton] = []
        self._color_btns:  List[UIButton] = []
        self._brush_btns:  List[UIButton] = []
        self._eraser_btns: List[UIButton] = []
        self._btn_pen:   UIButton
        self._btn_erase: UIButton
        self._btn_mode_1h: UIButton
        self._btn_mode_2h: UIButton

        self._build_toolbar()

    # ── Layout ───────────────────────────────────────────────

    def _build_toolbar(self):
        BH = 46
        BY = (TOOLBAR_H - BH) // 2
        x  = 8

        def add(b: UIButton) -> UIButton:
            self._buttons.append(b)
            return b

        # ── Tool buttons ──────────────────────────────────────
        self._btn_pen   = add(UIButton(x, BY, 60, BH, "Pen",    value=Tool.PEN));    x += 68
        self._btn_erase = add(UIButton(x, BY, 70, BH, "Eraser", value=Tool.ERASER)); x += 80

        x += 4
        self._btn_mode_1h = add(UIButton(x, BY, 44, BH, "1H", value=("mode", "1h"))); x += 50
        self._btn_mode_2h = add(UIButton(x, BY, 44, BH, "2H", value=("mode", "2h"))); x += 52

        x += 6

        # ── Colour swatches ───────────────────────────────────
        SW = 38; SY = (TOOLBAR_H - SW) // 2
        for cname, bgr in PALETTE.items():
            btn = UIButton(x, SY, SW, SW, cname, value=bgr, swatch=bgr)
            self._color_btns.append(btn)
            self._buttons.append(btn)
            x += SW + 5
        x += 6

        # ── Brush size ────────────────────────────────────────
        for sz in BRUSH_SIZES:
            btn = UIButton(x, BY+4, 30, BH-8, str(sz), value=("brush", sz))
            self._brush_btns.append(btn)
            self._buttons.append(btn)
            x += 36
        x += 6

        # ── Eraser size ───────────────────────────────────────
        for sz, lbl in zip(ERASER_SIZES, ["S", "M", "L", "XL"]):
            btn = UIButton(x, BY+4, 26, BH-8, lbl, value=("eraser", sz))
            self._eraser_btns.append(btn)
            self._buttons.append(btn)
            x += 32
        x += 6

        # ── Action buttons ────────────────────────────────────
        add(UIButton(x, BY, 62, BH, "Clear", value="clear")); x += 70
        add(UIButton(x, BY, 62, BH, "Save",  value="save"));  x += 70
        add(UIButton(x, BY, 62, BH, "Undo",  value="undo"));  x += 70
        add(UIButton(x, BY, 58, BH, "Z-",    value="zoom_out")); x += 66
        add(UIButton(x, BY, 58, BH, "Z+",    value="zoom_in"))

    # ── Tick ─────────────────────────────────────────────────

    def update(self, cursor: Optional[Tuple[int, int]]) -> List[str]:
        """
        Update all buttons and return a list of action strings fired
        this frame (e.g. ["clear"], ["save"], etc.).

        Pass cursor=None when the finger is outside the toolbar zone
        so buttons properly reset their hover state.
        """
        actions: List[str] = []
        for btn in self._buttons:
            if btn.update(cursor):
                act = self._handle(btn)
                if act:
                    actions.append(act)
        self._sync_selected()
        return actions

    def click(self, cursor: Tuple[int, int]) -> Optional[str]:
        """
        Immediate mouse/trackpad click activation for toolbar items.
        """
        for btn in self._buttons:
            if btn.hit(*cursor):
                act = self._handle(btn)
                self._sync_selected()
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

        elif isinstance(v, tuple) and len(v) == 2 and v[0] == "brush":
            self.brush_size = v[1]
            self.notify(f"Brush size: {v[1]}")

        elif isinstance(v, tuple) and len(v) == 2 and v[0] == "eraser":
            self.eraser_size = v[1]
            self.notify(f"Eraser size: {v[1]}")

        elif isinstance(v, tuple) and len(v) == 2 and v[0] == "mode":
            mode = GestureInputMode.ONE_HAND if v[1] == "1h" else GestureInputMode.TWO_HAND
            if mode != self.gesture_mode:
                self.gesture_mode = mode
                lbl = "One-hand mode" if mode == GestureInputMode.ONE_HAND else "Two-hand mode"
                self.notify(lbl)

        elif isinstance(v, tuple) and len(v) == 3:
            # colour swatch: value is the BGR tuple
            self.color      = v
            self.color_name = btn.label
            self.notify(f"Colour: {btn.label}")

        elif v in ("clear", "save", "undo", "zoom_in", "zoom_out"):
            return v

        return None

    def _sync_selected(self):
        self._btn_pen.selected   = (self.tool == Tool.PEN)
        self._btn_erase.selected = (self.tool == Tool.ERASER)
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
        """Draw the full toolbar onto `frame` in-place (BGR)."""
        # Semi-transparent background
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (self.fw, TOOLBAR_H), (16, 16, 24), -1)
        cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
        cv2.line(frame, (0, TOOLBAR_H), (self.fw, TOOLBAR_H), (80, 80, 80), 1)

        # Section micro-labels
        def lbl(t, px):
            cv2.putText(frame, t, (px, TOOLBAR_H - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (88, 88, 100), 1)
        lbl("TOOLS",   8)
        lbl("MODE",    160)
        lbl("COLOURS", 270)
        lbl("BRUSH",   620)
        lbl("ERASER",  805)
        lbl("ACTIONS", 940)

        for btn in self._buttons:
            btn.draw(frame)

        # Current-colour indicator dot (top-right corner)
        cr = (self.fw - 18, 18)
        cv2.circle(frame, cr, 11, self.color, -1)
        cv2.circle(frame, cr, 11, (105, 105, 105), 1)

        # Notification banner
        if self._notif and time.time() < self._notif_end:
            self._draw_notif(frame)

    def _draw_notif(self, frame: np.ndarray):
        msg  = self._notif
        font = cv2.FONT_HERSHEY_SIMPLEX; fsc = 0.62; tk = 2
        (tw, ht), _ = cv2.getTextSize(msg, font, fsc, tk)
        pad = 12
        nx  = (self.fw - tw) // 2
        ny  = TOOLBAR_H + 38
        cv2.rectangle(frame, (nx-pad, ny-ht-pad//2), (nx+tw+pad, ny+pad//2), (10,10,18), -1)
        cv2.rectangle(frame, (nx-pad, ny-ht-pad//2), (nx+tw+pad, ny+pad//2), (68,178,255), 1)
        cv2.putText(frame, msg, (nx, ny), font, fsc, (68, 200, 255), tk)
