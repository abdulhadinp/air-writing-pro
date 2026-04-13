"""
main.py
───────
Air Writing Pro — main entry point and application orchestrator.

Responsibilities
────────────────
• Camera lifecycle (open → loop → release)
• Frame capture + mirroring
• Subsystem wiring (tracker → gesture → canvas / ui)
• Cursor EMA smoothing
• HUD rendering
• Keyboard shortcut handling
"""

import cv2
import time
import math
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple

from hand_tracker      import HandTracker
from gesture_controller import Gesture, GestureController
from drawing_canvas    import DrawingCanvas
from ui_manager        import TOOLBAR_H, UI_ACTIVE_MARGIN, GestureInputMode, Tool, UIManager


# ── Tuning ───────────────────────────────────────────────────
CURSOR_ALPHA_SLOW = 0.20   # stronger smoothing for tiny movements
CURSOR_ALPHA_FAST = 0.72   # faster response for larger movements
CURSOR_FAST_PX    = 35.0   # movement (px/frame) where smoothing becomes responsive
CLEAR_COOLDOWN  = 2.5    # seconds between successive palm-clear gestures
DUAL_UNDO_COOLDOWN = 1.0
FRAME_RETRY_MS  = 30     # sleep ms on failed capture


class AirWritingPro:
    """
    Top-level application class.
    Instantiate once, then call `.run()`.
    """

    def __init__(self):
        # ── Camera ──────────────────────────────────────────
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Cannot open camera.\n"
                "macOS: System Preferences → Privacy & Security → Camera → "
                "allow your terminal / IDE."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self._cap.set(cv2.CAP_PROP_FPS,            30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,      1)   # low-latency

        ret, probe = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read initial frame from camera.")
        self._fh, self._fw = probe.shape[:2]

        # ── Subsystems ──────────────────────────────────────
        self._tracker = HandTracker()
        self._gesture = GestureController(toolbar_height=TOOLBAR_H)
        self._canvas  = DrawingCanvas(self._fw, self._fh)
        self._ui      = UIManager(self._fw, self._fh)

        # ── Cursor smoothing ─────────────────────────────────
        self._cursor_ema: Optional[Tuple[float, float]] = None
        self._pending_mouse_actions: list[str] = []

        # ── Stroke / erase state flags ───────────────────────
        self._stroke_active = False
        self._erase_active  = False

        # ── Pan tracking ─────────────────────────────────────
        self._pan_prev: Optional[Tuple[int, int]] = None

        # ── Clear-gesture debounce ───────────────────────────
        self._last_clear = 0.0
        self._last_dual_undo = 0.0
        self._dual_zoom_prev_dist: Optional[float] = None
        self._dual_pan_prev_center: Optional[Tuple[int, int]] = None

        # ── FPS counter ──────────────────────────────────────
        self._fps_buf: deque = deque(maxlen=30)
        self._last_t = time.time()

        self._print_welcome()

    # ════════════════════════════════════════════════════════
    #  Welcome
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _print_welcome():
        print()
        print("╔════════════════════════════════════════════╗")
        print("║       Air Writing Pro  —  Ready            ║")
        print("╠════════════════════════════════════════════╣")
        print("║  Gestures:                                 ║")
        print("║    Index only       → Draw                 ║")
        print("║    Index + Middle   → Erase                ║")
        print("║    Open palm        → Clear canvas         ║")
        print("║    4 fingers        → Pan canvas           ║")
        print("║    Finger on HUD    → Select tool/colour   ║")
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

    # ════════════════════════════════════════════════════════
    #  Internal helpers
    # ════════════════════════════════════════════════════════

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
            self._canvas.clear(); self._ui.notify("Canvas cleared!")
        elif act == "save":
            fn = self._canvas.save()
            self._ui.notify(f"Saved: {fn}"); print(f"[Save] {fn}")
        elif act == "undo":
            self._canvas.undo(); self._ui.notify("Undo!")
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

    def _pick_primary_hand(self, hands: list[dict]) -> Optional[dict]:
        if not hands:
            return None
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

    def _handle_dual_hand_gestures(self, hands: list[dict]) -> Optional[str]:
        if len(hands) < 2:
            self._reset_dual_trackers()
            return None

        hands2 = sorted(hands, key=lambda h: int(h["center"][0]))
        h1, h2 = hands2[0], hands2[-1]
        s1 = h1["states"]
        s2 = h2["states"]
        p1 = h1["index_tip"]
        p2 = h2["index_tip"]

        if not isinstance(s1, dict) or not isinstance(s2, dict):
            self._reset_dual_trackers()
            return None

        if self._is_open_palm(s1) and self._is_open_palm(s2):
            self._try_clear()
            self._reset_dual_trackers()
            return "2H CLEAR"

        now = time.time()
        if self._is_fist(s1) and self._is_fist(s2):
            if now - self._last_dual_undo > DUAL_UNDO_COOLDOWN:
                self._canvas.undo()
                self._ui.notify("Undo (2H)")
                self._last_dual_undo = now
            self._reset_dual_trackers()
            return "2H UNDO"

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

    # ════════════════════════════════════════════════════════
    #  HUD rendering
    # ════════════════════════════════════════════════════════

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

        # Gesture — bottom-centre
        _gmap = {
            Gesture.IDLE:     ("IDLE",     (110, 110, 110)),
            Gesture.DRAW:     ("DRAWING",  (60,  245, 60)),
            Gesture.ERASE:    ("ERASING",  (60,  100, 255)),
            Gesture.CLEAR:    ("CLEAR!",   (40,  40,  255)),
            Gesture.PAN:      ("PANNING",  (255, 195, 40)),
            Gesture.UI_HOVER: ("UI MODE",  (255, 120, 40)),
        }
        glbl, gcol = _gmap.get(gesture, ("?", (180, 180, 180)))
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

        # Pan offset
        if self._canvas.pan_x != 0 or self._canvas.pan_y != 0 or abs(self._canvas.zoom - 1.0) > 1e-3:
            ps = (
                f"Pan  dx={self._canvas.pan_x:+d}  dy={self._canvas.pan_y:+d}    "
                f"Zoom {self._canvas.zoom:.2f}x"
            )
            cv2.putText(frame, ps, (10, fh - 34), font, 0.38, (155, 155, 88), 1)

        # Cursor visualisation
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
                cv2.line(frame, (cx-5, cy),  (cx+5, cy),  (190, 190, 190), 1)
                cv2.line(frame, (cx, cy-5),  (cx, cy+5),  (190, 190, 190), 1)
            elif gesture == Gesture.UI_HOVER:
                cv2.circle(frame, (cx, cy), 10, (255, 195, 40), -1)
                cv2.circle(frame, (cx, cy), 13, (255, 255, 255), 2)
            elif gesture == Gesture.PAN:
                cv2.drawMarker(frame, (cx, cy), (255, 195, 40),
                               cv2.MARKER_CROSS, 22, 2)
            else:
                cv2.circle(frame, (cx, cy), 5, (190, 190, 190), -1)

        # Clear overlay
        if gesture == Gesture.CLEAR:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (fw, fh), (0, 0, 50), -1)
            cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)
            msg = "OPEN PALM  =  CLEAR CANVAS"
            tw  = cv2.getTextSize(msg, font, 0.95, 2)[0][0]
            cv2.putText(frame, msg, ((fw - tw) // 2, fh // 2),
                        font, 0.95, (50, 50, 255), 2)

    # ════════════════════════════════════════════════════════
    #  Main loop
    # ════════════════════════════════════════════════════════

    def run(self):
        cv2.namedWindow("Air Writing Pro", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Air Writing Pro", self._fw, self._fh)
        cv2.setMouseCallback("Air Writing Pro", self._on_mouse)

        while True:
            # ── Capture ──────────────────────────────────────
            ret, frame = self._cap.read()
            if not ret:
                print("[WARN] Dropped frame – retrying …")
                time.sleep(FRAME_RETRY_MS / 1000)
                continue
            frame = cv2.flip(frame, 1)     # mirror for natural interaction

            # ── Hand tracking ────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._tracker.process(rgb)

            hands = self._tracker.hand_infos(self._fw, self._fh)
            primary = self._pick_primary_hand(hands)
            fstates = primary["states"] if primary else None
            raw_pos = primary["index_tip"] if primary else None
            cursor  = self._smooth_cursor(raw_pos) if raw_pos else None

            # ── Gesture classification ───────────────────────
            gesture = self._gesture.update(fstates, cursor)
            dual_status: Optional[str] = None
            if self._ui.gesture_mode == GestureInputMode.TWO_HAND:
                dual_status = self._handle_dual_hand_gestures(hands)
            else:
                self._reset_dual_trackers()

            # ── UI update ────────────────────────────────────
            ui_cur  = cursor if (cursor and cursor[1] < (TOOLBAR_H + UI_ACTIVE_MARGIN)) else None
            ui_acts = self._ui.update(ui_cur)
            if self._pending_mouse_actions:
                ui_acts.extend(self._pending_mouse_actions)
                self._pending_mouse_actions.clear()

            for act in ui_acts:
                self._handle_ui_action(act)

            # ── Canvas operations ────────────────────────────
            in_zone = bool(cursor and cursor[1] >= (TOOLBAR_H + UI_ACTIVE_MARGIN))
            dual_active = dual_status is not None

            if in_zone and cursor is not None and not dual_active:
                eff_tool = self._ui.tool
                is_draw  = (gesture == Gesture.DRAW   and eff_tool == Tool.PEN)
                is_erase = (gesture == Gesture.ERASE) or \
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
                        self._canvas.begin_stroke()
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

                # Clear ───────────────────────────────────────
                if gesture == Gesture.CLEAR:
                    self._try_clear()

            else:
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

            # ── Keyboard ─────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'):
                print("Quitting …"); break
            elif key == ord('c'):
                self._canvas.clear();   self._ui.notify("Canvas cleared!")
            elif key == ord('u'):
                self._canvas.undo();    self._ui.notify("Undo!")
            elif key == ord('s'):
                fn = self._canvas.save()
                self._ui.notify(f"Saved: {fn}"); print(f"[Save] {fn}")
            elif key in (ord('+'), ord('=')):
                if self._canvas.zoom_by(+0.15):
                    self._ui.notify(f"Zoom: {self._canvas.zoom:.2f}x")
            elif key in (ord('-'), ord('_')):
                if self._canvas.zoom_by(-0.15):
                    self._ui.notify(f"Zoom: {self._canvas.zoom:.2f}x")
            elif key == ord('0'):
                self._canvas.reset_view(); self._ui.notify("View reset")

        # ── Cleanup ──────────────────────────────────────────
        self._cap.release()
        cv2.destroyAllWindows()
        print("Air Writing Pro closed.")


# ════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        AirWritingPro().run()
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}\n")
    except KeyboardInterrupt:
        print("\nInterrupted — bye!")
