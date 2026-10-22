import json
import math
import os
import threading
import traceback
from typing import Callable, List, Optional

import cv2
import numpy as np

from face_detection import Face, FaceTracker

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "filters")

class Filter:
    """
    A live view / photo filter. apply() may modify the frame in place and returns the result.
    """
    kind = "none"
    needs_faces = False

    def __init__(self, filter_id: str, name: str):
        self.id = filter_id
        self.name = name

    def apply(self, frame: np.ndarray, faces: List[Face]) -> np.ndarray:
        return frame

# ---------------------------------------------------------------------------
# AR stickers
# ---------------------------------------------------------------------------

# Points a sticker layer can be pinned to
ANCHORS = {
    "eyes": lambda face: face.eye_center,
    "nose": lambda face: face.nose_tip,
    "upper_lip": lambda face: face.upper_lip,
    "forehead": lambda face: face.forehead,
    "chin": lambda face: face.chin,
}

class StickerLayer:
    def __init__(self, spec: dict):
        path = os.path.join(ASSETS_DIR, spec["image"])
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"Sticker image {path} must be a PNG with an alpha channel")
        if spec.get("anchor", "eyes") not in ANCHORS:
            raise ValueError(f"Unknown sticker anchor '{spec.get('anchor')}' in {spec['image']}")
        self.image = image
        self.anchor = ANCHORS[spec.get("anchor", "eyes")]
        # Sticker width, as a multiple of the distance between the eyes
        self.width = float(spec.get("width", 1.0))
        # Shift from the anchor in eye-distance units: x along the eye line, y towards the chin
        self.offset = np.array(spec.get("offset", [0.0, 0.0]), dtype=np.float32)
        # Point of the sticker placed on the anchor, as a fraction of its width/height
        self.pivot = np.array(spec.get("pivot", [0.5, 0.5]), dtype=np.float32)
        self.rotate = bool(spec.get("rotate", True))

    def draw(self, frame: np.ndarray, face: Face):
        eye_distance = face.eye_distance
        if eye_distance < 1:
            return
        sticker_h, sticker_w = self.image.shape[:2]
        scale = self.width * eye_distance / sticker_w
        angle = face.roll if self.rotate else 0.0
        cos, sin = math.cos(angle), math.sin(angle)
        rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)

        target = self.anchor(face) + rotation @ (self.offset * eye_distance)
        pivot = self.pivot * np.array([sticker_w, sticker_h], dtype=np.float32)
        linear = rotation * scale
        matrix = np.hstack([linear, (target - linear @ pivot)[:, None]])

        # Only warp the region of the frame the sticker covers
        corners = np.array([[0, 0], [sticker_w, 0], [0, sticker_h], [sticker_w, sticker_h]], dtype=np.float32)
        placed = corners @ linear.T + matrix[:, 2]
        frame_h, frame_w = frame.shape[:2]
        x0, y0 = np.floor(placed.min(axis=0)).astype(int)
        x1, y1 = np.ceil(placed.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, frame_w), min(y1, frame_h)
        if x1 <= x0 or y1 <= y0:
            return

        matrix[:, 2] -= (x0, y0)
        warped = cv2.warpAffine(self.image, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        alpha = warped[:, :, 3:].astype(np.float32) / 255.0
        roi = frame[y0:y1, x0:x1]
        roi[:] = (warped[:, :, :3] * alpha + roi * (1.0 - alpha)).astype(np.uint8)

class StickerFilter(Filter):
    kind = "sticker"
    needs_faces = True

    def __init__(self, filter_id: str, name: str, layers: List[dict]):
        super().__init__(filter_id, name)
        self.layers = [StickerLayer(layer) for layer in layers]

    def apply(self, frame, faces):
        for face in faces:
            for layer in self.layers:
                layer.draw(frame, face)
        return frame

# ---------------------------------------------------------------------------
# Face effects
# ---------------------------------------------------------------------------

def _face_region(frame: np.ndarray, face: Face, feather: float = 0.0):
    """
    Returns (x0, y0, x1, y1, mask) for the face oval clipped to the frame, where mask is a
    float32 (h, w) weight map, optionally feathered by a fraction of the eye distance.
    """
    frame_h, frame_w = frame.shape[:2]
    margin = int(feather * face.eye_distance)
    x, y, w, h = face.bbox
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + w + margin, frame_w), min(y + h + margin, frame_h)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [face.oval - (x0, y0)], 255, lineType=cv2.LINE_AA)
    if margin:
        # Stack blur costs the same for any kernel size, unlike a large Gaussian
        kernel = margin * 2 + 1
        mask = cv2.stackBlur(mask, (kernel, kernel))
    return x0, y0, x1, y1, mask.astype(np.float32) / 255.0

class MaskedFaceEffect(Filter):
    """
    Replaces the face oval with a processed version of itself.
    """
    kind = "effect"
    needs_faces = True

    def __init__(self, filter_id: str, name: str, process: Callable[[np.ndarray, Face], np.ndarray], feather: float = 0.0):
        super().__init__(filter_id, name)
        self.process = process
        self.feather = feather

    def apply(self, frame, faces):
        for face in faces:
            region = _face_region(frame, face, self.feather)
            if region is None:
                continue
            x0, y0, x1, y1, mask = region
            roi = frame[y0:y1, x0:x1]
            roi[:] = cv2.blendLinear(self.process(roi, face), roi, mask, 1.0 - mask)
        return frame

def _pixelate(roi: np.ndarray, face: Face) -> np.ndarray:
    h, w = roi.shape[:2]
    blocks = 12
    small = cv2.resize(roi, (blocks, max(1, round(blocks * h / w))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def _blur(roi: np.ndarray, face: Face) -> np.ndarray:
    # Blur a tiny copy and scale it back up: the same look as a huge kernel at a fraction of the cost
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (32, max(1, round(32 * h / w))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def _smooth(roi: np.ndarray, face: Face) -> np.ndarray:
    # Bilateral filtering is costly on large faces, so smooth a reduced copy and keep
    # the original's fine detail with a light blend
    h, w = roi.shape[:2]
    factor = min(1.0, 320 / max(w, 1))
    small = cv2.resize(roi, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else roi
    smoothed = cv2.bilateralFilter(small, 9, 60, 9)
    if factor < 1:
        smoothed = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(smoothed, 0.75, roi, 0.25, 0)

class BigEyesFilter(Filter):
    kind = "effect"
    needs_faces = True

    def __init__(self, filter_id: str, name: str, strength: float = 0.45, radius: float = 0.5):
        super().__init__(filter_id, name)
        self.strength = strength
        # Warp radius, as a multiple of the eye distance
        self.radius = radius

    def apply(self, frame, faces):
        frame_h, frame_w = frame.shape[:2]
        for face in faces:
            r = self.radius * face.eye_distance
            if r < 2:
                continue
            for cx, cy in (face.right_eye, face.left_eye):
                x0, y0 = max(int(cx - r), 0), max(int(cy - r), 0)
                x1, y1 = min(int(cx + r) + 1, frame_w), min(int(cy + r) + 1, frame_h)
                if x1 <= x0 or y1 <= y0:
                    continue
                xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
                dx, dy = xs - cx, ys - cy
                dist = np.sqrt(dx * dx + dy * dy) / r
                # Sample closer to the center inside the radius, which magnifies the eye;
                # k reaches 1 at the edge so the warp blends seamlessly
                k = np.where(dist < 1.0, 1.0 - self.strength * (1.0 - dist * dist), 1.0).astype(np.float32)
                map_x = cx + dx * k
                map_y = cy + dy * k
                frame[y0:y1, x0:x1] = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return frame

# ---------------------------------------------------------------------------
# Whole-frame color
# ---------------------------------------------------------------------------

# Sepia in BGR order (rows produce B, G, R from B, G, R inputs)
SEPIA = np.array([[0.131, 0.534, 0.272],
                  [0.168, 0.686, 0.349],
                  [0.189, 0.769, 0.393]], dtype=np.float32)

class ColorFilter(Filter):
    kind = "color"

    def __init__(self, filter_id: str, name: str, transform: Callable[[np.ndarray], np.ndarray]):
        super().__init__(filter_id, name)
        self.transform = transform

    def apply(self, frame, faces):
        return self.transform(frame)

def _grayscale(frame):
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

def _sepia(frame):
    return cv2.transform(frame, SEPIA)

_vignettes: dict = {}
_vignette_lock = threading.Lock()

# Sepia scaled down with lifted blacks, as a 3x4 affine color transform
VINTAGE = np.hstack([SEPIA * 0.85, np.full((3, 1), 25, dtype=np.float32)])

def _vignette(shape) -> np.ndarray:
    """
    Cached uint8 (h, w, 3) brightness map, 255 in the center and darker towards the corners.
    """
    h, w = shape[:2]
    with _vignette_lock:
        if (h, w) not in _vignettes:
            ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
            xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
            falloff = np.clip(1.0 - 0.45 * (xs * xs + ys * ys), 0.0, 1.0)
            _vignettes[(h, w)] = cv2.merge([(falloff * 255).astype(np.uint8)] * 3)
        return _vignettes[(h, w)]

def _vintage(frame):
    # Sepia, faded blacks and darkened corners
    return cv2.multiply(cv2.transform(frame, VINTAGE), _vignette(frame.shape), scale=1 / 255)

def _channel_lut(b: float, g: float, r: float) -> np.ndarray:
    # Gains below 1 scale the channel down; above 1 they lift it towards white without clipping
    ramp = np.arange(256, dtype=np.float32)
    channels = [ramp * gain if gain <= 1 else ramp + (255 - ramp) * (gain - 1) for gain in (b, g, r)]
    return np.stack(channels, axis=-1).astype(np.uint8)[None, :, :]

WARM_LUT = _channel_lut(0.85, 1.0, 1.12)
COOL_LUT = _channel_lut(1.12, 1.0, 0.88)

def _warm(frame):
    return cv2.LUT(frame, WARM_LUT)

def _cool(frame):
    return cv2.LUT(frame, COOL_LUT)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _builtin_filters() -> List[Filter]:
    return [
        Filter("none", "None"),
        MaskedFaceEffect("pixelate", "Pixelate", _pixelate),
        MaskedFaceEffect("blur", "Blur Face", _blur, feather=0.15),
        MaskedFaceEffect("smooth", "Beauty", _smooth, feather=0.2),
        BigEyesFilter("big_eyes", "Big Eyes"),
        ColorFilter("grayscale", "Black & White", _grayscale),
        ColorFilter("sepia", "Sepia", _sepia),
        ColorFilter("vintage", "Vintage", _vintage),
        ColorFilter("warm", "Warm", _warm),
        ColorFilter("cool", "Cool", _cool),
    ]

def _load_stickers(manifest_path: str) -> List[Filter]:
    """
    Loads sticker filters from the manifest; each entry is
    {"id", "name", "layers": [{"image", "anchor", "width", "offset", "pivot", "rotate"}]}.
    """
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path) as f:
        manifest = json.load(f)
    stickers = []
    for entry in manifest.get("stickers", []):
        try:
            stickers.append(StickerFilter(entry["id"], entry.get("name", entry["id"]), entry["layers"]))
        except (KeyError, ValueError) as e:
            print(f"Warning: skipping sticker filter {entry.get('id')}: {e}")
    return stickers

class FilterEngine:
    """
    Holds the available filters and the one currently active for the booth.
    apply_video() is for consecutive live view frames (single producer thread);
    apply_image() is for still photos and may be called from any thread.
    """
    def __init__(self, default_filter: Optional[str] = None, manifest_path: Optional[str] = None):
        filters = _builtin_filters()
        # Stickers come right after "none" so they lead the list in the booth UI
        filters[1:1] = _load_stickers(manifest_path or os.path.join(ASSETS_DIR, "filters.json"))
        self._filters = {f.id: f for f in filters}
        self._active = "none"
        if default_filter:
            try:
                self.set_active(default_filter)
            except ValueError as e:
                print(f"Warning: {e}; starting with no filter")

        # Trackers load the model lazily, so the service starts fast and "none" costs nothing
        self._stream_tracker: Optional[FaceTracker] = None
        self._image_tracker: Optional[FaceTracker] = None
        self._image_lock = threading.Lock()
        self._failed: set = set()

    @property
    def active(self) -> str:
        return self._active

    @property
    def is_active(self) -> bool:
        return self._active != "none"

    def list(self) -> List[dict]:
        return [{"id": f.id, "name": f.name, "kind": f.kind} for f in self._filters.values()]

    def set_active(self, filter_id: str):
        if filter_id not in self._filters:
            raise ValueError(f"Unknown filter '{filter_id}'")
        self._active = filter_id

    def apply_video(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies the active filter to a live view frame. Blocking; run in a thread.
        """
        f = self._filters[self._active]
        if f.id == "none":
            return frame
        return self._apply(f, frame, lambda: self._detect_stream(frame) if f.needs_faces else [])

    def apply_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies the active filter to a still photo. Blocking; run in a thread.
        """
        f = self._filters[self._active]
        if f.id == "none":
            return frame
        with self._image_lock:
            return self._apply(f, frame, lambda: self._detect_image(frame) if f.needs_faces else [])

    def _detect_stream(self, frame):
        if self._stream_tracker is None:
            self._stream_tracker = FaceTracker(mode="stream")
        return self._stream_tracker.detect(frame)

    def _detect_image(self, frame):
        if self._image_tracker is None:
            self._image_tracker = FaceTracker(mode="image")
        return self._image_tracker.detect(frame)

    def _apply(self, f: Filter, frame: np.ndarray, detect: Callable[[], List[Face]]) -> np.ndarray:
        # A broken filter must never take down the live view or lose a photo:
        # fall back to the unfiltered frame and report the error once per filter
        try:
            return f.apply(frame, detect())
        except Exception:
            if f.id not in self._failed:
                self._failed.add(f.id)
                print(f"Warning: filter '{f.id}' failed, showing unfiltered frames")
                traceback.print_exc()
            return frame

    def close(self):
        for tracker in (self._stream_tracker, self._image_tracker):
            if tracker is not None:
                tracker.close()
