"""
drawing_canvas.py
─────────────────
The drawing surface layer.

• Completely separate from the camera feed (composited at render time)
• Smooth stroke interpolation with sub-pixel density
• Speed-aware precision: slow movement → finer strokes
• Weighted moving-average jitter reduction
• Full undo stack (up to MAX_UNDO_STEPS snapshots)
• Canvas panning via integer pixel offsets
• PNG export with auto-timestamped filenames
"""

import cv2
import math
import numpy as np
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple

# ── Tuning constants ─────────────────────────────────────────
STROKE_SMOOTH   = 5     # weighted-MA window length
MAX_UNDO_STEPS  = 50    # maximum snapshots kept in RAM


class DrawingCanvas:
    """
    A NumPy-backed drawing layer that composites onto a camera frame.

    Coordinates
    ───────────
    All public methods accept *screen-space* coordinates.
    `pan_x` / `pan_y` shift the canvas relative to the screen so that
    panning doesn't move existing content — it moves the viewport.
    """

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height

        self._cvs: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        self._hist: List[np.ndarray] = []

        # Active-stroke state (reset between strokes)
        self._prev_pt: Optional[Tuple[int, int]] = None
        self._sx: deque = deque(maxlen=STROKE_SMOOTH)
        self._sy: deque = deque(maxlen=STROKE_SMOOTH)

        # Viewport pan offset
        self.pan_x: int = 0
        self.pan_y: int = 0
        self.zoom: float = 1.0
        self._min_zoom: float = 0.6
        self._max_zoom: float = 3.0

    # ════════════════════════════════════════════════════════
    #  History / Undo
    # ════════════════════════════════════════════════════════

    def _push_snapshot(self):
        """Save current canvas state (oldest entry dropped at MAX_UNDO_STEPS)."""
        if len(self._hist) >= MAX_UNDO_STEPS:
            self._hist.pop(0)
        self._hist.append(self._cvs.copy())

    def undo(self):
        """Restore the previous canvas state."""
        if self._hist:
            self._cvs = self._hist.pop()
            self._reset_stroke()

    def clear(self):
        """Erase the entire canvas (undoable)."""
        self._push_snapshot()
        self._cvs[:] = 0
        self._reset_stroke()

    # ════════════════════════════════════════════════════════
    #  Stroke lifecycle
    # ════════════════════════════════════════════════════════

    def begin_stroke(self):
        """
        Call when a new draw/erase stroke begins.
        Saves an undo snapshot so the whole stroke can be undone at once.
        """
        self._push_snapshot()
        self._reset_stroke()

    def end_stroke(self):
        """Call when the stroke ends (finger lifts / gesture changes)."""
        self._reset_stroke()

    # ════════════════════════════════════════════════════════
    #  Drawing operations
    # ════════════════════════════════════════════════════════

    def draw_at(
        self,
        sx: int, sy: int,
        color: Tuple[int, int, int],
        base_size: int,
    ):
        """
        Draw a smooth, continuous stroke segment ending at screen (sx, sy).

        Algorithm
        ─────────
        1. Convert to canvas space (subtract pan offset)
        2. Apply weighted moving average for jitter reduction
        3. Compute speed-adjusted brush radius
        4. Interpolate between previous and current point at 1 px steps
           (guarantees no gaps even at high cursor velocity)
        """
        cx, cy = self._to_canvas(sx, sy)
        cx, cy = self._smooth(cx, cy)
        size   = self._speed_size(cx, cy, base_size)

        if self._prev_pt is not None:
            px, py = self._prev_pt
            dist   = math.hypot(cx - px, cy - py)
            steps  = max(1, int(dist))
            for k in range(steps + 1):
                t  = k / steps
                ix = int(px + t * (cx - px))
                iy = int(py + t * (cy - py))
                self._safe_circle((ix, iy), size, color)
        else:
            self._safe_circle((cx, cy), size, color)

        self._prev_pt = (cx, cy)

    def erase_at(self, sx: int, sy: int, eraser_size: int):
        """
        Erase a circular region centred at screen (sx, sy).
        Interpolated like draw_at — no eraser gaps.
        """
        cx, cy = self._to_canvas(sx, sy)
        cx, cy = self._smooth(cx, cy)

        if self._prev_pt is not None:
            px, py = self._prev_pt
            dist   = math.hypot(cx - px, cy - py)
            steps  = max(1, int(dist))
            for k in range(steps + 1):
                t  = k / steps
                ix = int(px + t * (cx - px))
                iy = int(py + t * (cy - py))
                self._safe_circle((ix, iy), eraser_size, (0, 0, 0))
        else:
            self._safe_circle((cx, cy), eraser_size, (0, 0, 0))

        self._prev_pt = (cx, cy)

    def pan(self, dx: int, dy: int):
        """Shift the viewport by (dx, dy) pixels."""
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

    # ════════════════════════════════════════════════════════
    #  Compositing
    # ════════════════════════════════════════════════════════

    def composite(self, camera_frame: np.ndarray) -> np.ndarray:
        """
        Blend the canvas layer over `camera_frame`.

        Canvas pixels with any content (luminance > threshold) fully
        replace the corresponding camera pixel — no alpha blending needed,
        giving crisp, opaque strokes.
        """
        result = camera_frame.copy()
        disp   = self._shifted_canvas()

        # Boolean mask: canvas has content here
        mask = np.any(disp > 12, axis=2)   # slight threshold vs. sensor noise
        result[mask] = disp[mask]
        return result

    # ════════════════════════════════════════════════════════
    #  Export
    # ════════════════════════════════════════════════════════

    def save(self, path: Optional[str] = None) -> str:
        """Save canvas as PNG. Auto-generates a timestamped filename if needed."""
        if path is None:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"air_writing_{ts}.png"
        cv2.imwrite(path, self._cvs)
        return path

    # ════════════════════════════════════════════════════════
    #  Private helpers
    # ════════════════════════════════════════════════════════

    def _reset_stroke(self):
        self._prev_pt = None
        self._sx.clear()
        self._sy.clear()

    def _to_canvas(self, sx: int, sy: int) -> Tuple[int, int]:
        """Screen → canvas coordinate conversion."""
        return (
            int(round((sx - self.pan_x) / self.zoom)),
            int(round((sy - self.pan_y) / self.zoom)),
        )

    def _smooth(self, x: int, y: int) -> Tuple[int, int]:
        """
        Linearly-weighted moving average.
        Recent samples have proportionally higher weight.
        """
        self._sx.append(x); self._sy.append(y)
        n = len(self._sx)
        w = np.arange(1, n + 1, dtype=float)
        w /= w.sum()
        return (
            int(np.dot(w, list(self._sx))),
            int(np.dot(w, list(self._sy))),
        )

    def _speed_size(self, cx: int, cy: int, base: int) -> int:
        """
        Precision-mode: reduce brush radius slightly when the cursor is slow.
        Helps produce fine detail when the user deliberately moves carefully.
        """
        if self._prev_pt is None:
            return base
        dist   = math.hypot(cx - self._prev_pt[0], cy - self._prev_pt[1])
        factor = max(0.55, min(1.0, dist / 18.0))
        return max(1, int(base * factor))

    def _safe_circle(
        self,
        pt: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int],
    ):
        """Draw a filled circle only if the centre is inside the canvas."""
        px, py = pt
        if 0 <= px < self.width and 0 <= py < self.height:
            cv2.circle(self._cvs, (px, py), radius, color, -1)

    def _shifted_canvas(self) -> np.ndarray:
        """Return a copy of the canvas transformed by pan + zoom."""
        z_w, z_h = self._zoomed_size()
        if abs(self.zoom - 1.0) < 1e-6:
            scaled = self._cvs
        else:
            interp = cv2.INTER_LINEAR if self.zoom >= 1.0 else cv2.INTER_AREA
            scaled = cv2.resize(self._cvs, (z_w, z_h), interpolation=interp)

        out = np.zeros_like(self._cvs)
        sx1 = max(0, -self.pan_x)
        sy1 = max(0, -self.pan_y)
        dx1 = max(0, self.pan_x)
        dy1 = max(0, self.pan_y)
        cw = min(self.width - dx1, scaled.shape[1] - sx1)
        ch = min(self.height - dy1, scaled.shape[0] - sy1)

        if cw > 0 and ch > 0:
            out[dy1:dy1+ch, dx1:dx1+cw] = scaled[sy1:sy1+ch, sx1:sx1+cw]

        return out

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
