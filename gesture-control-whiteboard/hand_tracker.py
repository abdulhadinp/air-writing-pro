"""
hand_tracker.py
───────────────
MediaPipe Hands wrapper.
Provides per-finger up/down state and pixel-space landmark positions.
"""

import os
import math
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple

_mp_solutions = getattr(mp, "solutions", None)

HAND_LANDMARKER_ENV = "MP_HAND_LANDMARKER_TASK"
HAND_LANDMARKER_URLS = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
)


class HandTracker:
    """
    Thin, stateless (per-frame) wrapper around MediaPipe Hands.

    Usage
    ─────
        tracker = HandTracker()
        # inside loop:
        tracker.process(rgb_frame)
        states = tracker.finger_states(fw, fh)
        pos    = tracker.index_tip(fw, fh)
        tracker.draw_landmarks(bgr_frame)
    """

    # ── MediaPipe landmark IDs ────────────────────────────────
    WRIST      = 0
    THUMB_CMC  = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
    INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
    MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
    RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
    PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

    def __init__(
        self,
        max_hands: int = 2,
        detect_conf: float = 0.72,
        track_conf: float  = 0.62,
        model_complexity: int = 0,
    ):
        if _mp_solutions is not None:
            self._mp_hands   = _mp_solutions.hands
            self._draw_utils = _mp_solutions.drawing_utils
            self._hands = self._mp_hands.Hands(
                static_image_mode        = False,
                max_num_hands            = max_hands,
                model_complexity         = model_complexity,
                min_detection_confidence = detect_conf,
                min_tracking_confidence  = track_conf,
            )
            self._connections = self._mp_hands.HAND_CONNECTIONS
        else:
            try:
                self._hands = mp.tasks.vision.HandLandmarker.create_from_options(
                    mp.tasks.vision.HandLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(
                            model_asset_path=self._find_hand_landmarker_task(),
                            delegate=mp.tasks.BaseOptions.Delegate.CPU,
                        ),
                        running_mode=mp.tasks.vision.RunningMode.IMAGE,
                        num_hands=max_hands,
                        min_hand_detection_confidence=detect_conf,
                        min_hand_presence_confidence=0.5,
                        min_tracking_confidence=track_conf,
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
        self.landmarks = None   # first hand landmarks (compat)
        self.detected  = False

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
        """
        Run MediaPipe inference on an RGB frame.
        Must be called once per frame before any other method.
        Returns the first hand's landmarks (or None).
        """
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
        # Thumb: tip must be further from palm centre than IP joint
        idx_mcp_x = lm[self.INDEX_MCP].x
        wrist_x   = lm[self.WRIST].x
        t_tip     = lm[self.THUMB_TIP]
        t_ip      = lm[self.THUMB_IP]

        if idx_mcp_x > wrist_x:      # right-hand geometry in mirror
            thumb_ext = t_tip.x < t_ip.x
        else:                         # left-hand geometry
            thumb_ext = t_tip.x > t_ip.x

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
            "thumb": thumb_ext,
            "index": idx_s > -0.08,
            "middle": mid_s > -0.03,
            "ring": rng_s > 0.02,
            "pinky": pky_s > 0.02,
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

    # ── Finger state ─────────────────────────────────────────

    def finger_states(self, fw: int, fh: int) -> Optional[Dict[str, bool]]:
        """
        Returns {'thumb','index','middle','ring','pinky'} → bool (extended?)
        or None if no hand is detected.

        Thumb  — horizontal comparison (works for mirrored feed on either hand)
        Others — vertical comparison: tip.y < pip.y → extended
        """
        lm = self._normalized_landmarks()
        if not lm:
            return None

        return self._finger_states_for(lm)

    # ── Landmark positions ───────────────────────────────────

    def tip_px(self, tip_id: int, fw: int, fh: int, hand_idx: int = 0) -> Optional[Tuple[int, int]]:
        """Convert a landmark to integer pixel coordinates."""
        lm = self._normalized_landmarks(hand_idx)
        if not lm:
            return None
        lm = lm[tip_id]
        return int(lm.x * fw), int(lm.y * fh)

    def index_tip(self, fw: int, fh: int, hand_idx: int = 0) -> Optional[Tuple[int, int]]:
        return self.tip_px(self.INDEX_TIP, fw, fh, hand_idx=hand_idx)

    def middle_tip(self, fw: int, fh: int, hand_idx: int = 0) -> Optional[Tuple[int, int]]:
        return self.tip_px(self.MIDDLE_TIP, fw, fh, hand_idx=hand_idx)

    def wrist_px(self, fw: int, fh: int, hand_idx: int = 0) -> Optional[Tuple[int, int]]:
        return self.tip_px(self.WRIST, fw, fh, hand_idx=hand_idx)

    # ── Rendering ────────────────────────────────────────────

    def draw_landmarks(self, bgr_frame: np.ndarray):
        """Render hand skeleton on the frame in-place (BGR)."""
        for hand_lm in self._hand_landmarks:
            self._draw_utils.draw_landmarks(
                bgr_frame,
                hand_lm,
                self._connections,
                self._draw_utils.DrawingSpec(color=(0, 230, 120), thickness=2, circle_radius=3),
                self._draw_utils.DrawingSpec(color=(0, 170, 90),  thickness=1),
            )
