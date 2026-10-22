import math
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = os.getenv("FACE_MODEL_PATH", os.path.join("models", "face_landmarker.task"))

# Frames are downscaled to this width before detection; landmarks are normalized so they
# map back onto the full-resolution frame (1080p live view or 4K photo alike)
DETECT_WIDTH = 640

# Weight of the newest landmarks when smoothing streamed frames (lower = steadier but laggier)
SMOOTHING = 0.6

# Streamed results older than this are treated as "no face", e.g. after detection was paused
STALE_RESULT_MS = 500

# Face mesh landmark indices (478-point model with irises). "Right"/"left" are the subject's,
# so on an unmirrored camera image the right eye appears on the image's left.
RIGHT_IRIS = 468
LEFT_IRIS = 473
NOSE_TIP = 1
PHILTRUM = 164
FOREHEAD = 10
CHIN = 152
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

_download_lock = threading.Lock()

def ensure_model(path: str = MODEL_PATH) -> str:
    """
    Downloads the face landmarker model on first use.
    """
    with _download_lock:
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            print(f"Downloading face landmarker model to {path}")
            tmp_path = path + ".part"
            urllib.request.urlretrieve(MODEL_URL, tmp_path)
            os.replace(tmp_path, path)
    return path

@dataclass
class Face:
    # (478, 2) landmark positions in frame pixels
    points: np.ndarray

    def point(self, index: int) -> np.ndarray:
        return self.points[index]

    @property
    def right_eye(self) -> np.ndarray:
        return self.points[RIGHT_IRIS]

    @property
    def left_eye(self) -> np.ndarray:
        return self.points[LEFT_IRIS]

    @property
    def eye_center(self) -> np.ndarray:
        return (self.right_eye + self.left_eye) / 2

    @property
    def eye_distance(self) -> float:
        return float(np.linalg.norm(self.left_eye - self.right_eye))

    @property
    def roll(self) -> float:
        """
        Head tilt in radians, as the angle of the eye line in image coordinates.
        """
        dx, dy = self.left_eye - self.right_eye
        return math.atan2(dy, dx)

    @property
    def nose_tip(self) -> np.ndarray:
        return self.points[NOSE_TIP]

    @property
    def upper_lip(self) -> np.ndarray:
        return self.points[PHILTRUM]

    @property
    def forehead(self) -> np.ndarray:
        return self.points[FOREHEAD]

    @property
    def chin(self) -> np.ndarray:
        return self.points[CHIN]

    @property
    def oval(self) -> np.ndarray:
        return self.points[FACE_OVAL].astype(np.int32)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """
        (x, y, w, h) of the face oval, in frame pixels (may extend past the frame).
        """
        return cv2.boundingRect(self.oval)

class FaceTracker:
    """
    Detects faces and their landmarks with MediaPipe Face Landmarker.

    mode="stream" is for live view: frames are submitted asynchronously and detect() returns
    the newest smoothed landmarks without waiting, so a slow model (~40 ms on CPU) never
    holds back the frame rate; frames arriving while the model is busy are skipped.
    mode="image" detects synchronously on each call, for still photos.
    An instance must only be used by one thread at a time.
    """
    def __init__(self, mode: str = "stream", max_faces: int = 4):
        if mode not in ("stream", "image"):
            raise ValueError(f"Invalid tracker mode '{mode}'. Expected 'stream' or 'image'")
        self._stream = mode == "stream"
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.LIVE_STREAM if self._stream else vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_result if self._stream else None,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp = -1
        # Newest streamed result in normalized coordinates, written by MediaPipe's callback thread
        self._result_lock = threading.Lock()
        self._latest: List[np.ndarray] = []
        self._latest_timestamp = -1

    def close(self):
        self._landmarker.close()

    def detect(self, frame: np.ndarray) -> List[Face]:
        height, width = frame.shape[:2]
        small = frame
        if width > DETECT_WIDTH:
            small = cv2.resize(frame, (DETECT_WIDTH, round(height * DETECT_WIDTH / width)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        scale = np.array([width, height], dtype=np.float32)

        if not self._stream:
            result = self._landmarker.detect(image)
            return [Face(points * scale) for points in _normalized_points(result)]

        # Live stream mode requires strictly increasing timestamps
        timestamp = max(int(time.monotonic() * 1000), self._last_timestamp + 1)
        self._last_timestamp = timestamp
        self._landmarker.detect_async(image, timestamp)
        with self._result_lock:
            if timestamp - self._latest_timestamp > STALE_RESULT_MS:
                return []
            return [Face(points * scale) for points in self._latest]

    def _on_result(self, result, image, timestamp_ms: int):
        with self._result_lock:
            previous = self._latest if timestamp_ms - self._latest_timestamp <= STALE_RESULT_MS else []
            self._latest = _smooth(_normalized_points(result), previous)
            self._latest_timestamp = timestamp_ms

def _normalized_points(result) -> List[np.ndarray]:
    return [np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float32) for landmarks in result.face_landmarks]

def _smooth(faces: List[np.ndarray], previous: List[np.ndarray]) -> List[np.ndarray]:
    """
    Blends each face with the nearest face from the previous result to reduce jitter.
    """
    smoothed = []
    unused = list(previous)
    for points in faces:
        center = points.mean(axis=0)
        if unused:
            nearest = min(unused, key=lambda prev: np.linalg.norm(prev.mean(axis=0) - center))
            # Only blend with a face that is plausibly the same person
            if np.linalg.norm(nearest.mean(axis=0) - center) < np.ptp(points[:, 0]):
                unused = [prev for prev in unused if prev is not nearest]
                points = SMOOTHING * points + (1 - SMOOTHING) * nearest
        smoothed.append(points)
    return smoothed

if __name__ == "__main__":
    # Debug preview: run filters on the webcam locally. n/p cycle filters, q quits.
    from filters import FilterEngine

    device = os.getenv("WEBCAM_DEVICE") or "0"
    cap = cv2.VideoCapture(int(device) if device.isdigit() else device)
    engine = FilterEngine()
    filter_ids = [f["id"] for f in engine.list()]
    index = filter_ids.index(engine.active)
    print("Press n/p to cycle filters, q to quit.")

    last = time.monotonic()
    fps = 0.0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = engine.apply_video(frame)

        now = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        cv2.putText(frame, f"{engine.active}  {fps:.0f} fps", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face filters", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("n"), ord("p")):
            index = (index + (1 if key == ord("n") else -1)) % len(filter_ids)
            engine.set_active(filter_ids[index])

    cap.release()
    cv2.destroyAllWindows()
