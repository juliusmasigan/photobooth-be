import asyncio
import collections
import os
import sys
import time
from contextlib import aclosing
from typing import Optional, Union
import cv2

import clip_encoder
from filters import FilterEngine

CAMERA_TYPES = ("dslr", "webcam")

# Number of frames to discard after opening a device, so auto-exposure/white balance can settle
WARMUP_FRAMES = 5

# Live view is kept at HD to stay light; webcam photos are taken at 4K
LIVE_VIEW_RESOLUTION = (1920, 1080)
PHOTO_RESOLUTION = (3840, 2160)

CLIP_KINDS = ("gif", "boomerang")
# Extra time allowed on top of a clip's length for the device to deliver its frames
CLIP_FRAME_TIMEOUT = 10

class CameraError(Exception):
    pass

def _timestamped_name(file_prefix: str) -> str:
    """
    Returns "<prefix>_<YYYYmmdd_HHMMSS>_<ms>", a file stem that won't clash between captures.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    return f"{file_prefix}_{timestamp}_{ms:03d}"

def _parse_device(value) -> Union[int, str]:
    """
    Accepts a device index (e.g. 0, "2") or a device path
    (e.g. "/dev/video0", "/dev/v4l/by-id/usb-...-video-index0").
    """
    if value is None or str(value).strip() == "":
        return 0
    value = str(value).strip()
    return int(value) if value.isdigit() else value

def _resample(recorded: list[tuple[float, bytes]], duration: float, fps: float) -> list[bytes]:
    """
    The device frame rate wobbles; picks the latest (timestamp, jpeg) frame at each tick
    of a steady fps clock over duration seconds.
    """
    start = recorded[0][0]
    frames: list[bytes] = []
    i = 0
    for k in range(max(2, round(duration * fps))):
        target = start + k / fps
        while i + 1 < len(recorded) and recorded[i + 1][0] <= target:
            i += 1
        frames.append(recorded[i][1])
    return frames

class CameraService:
    def __init__(self, camera_type: str = "dslr", live_view_device=0, webcam_device=None, live_view_fps: int = 30,
                 filters: Optional[FilterEngine] = None, gif_frames: int = 6, gif_interval_ms: int = 500,
                 boomerang_duration_ms: int = 1500, boomerang_fps: int = 20, clip_gif_width: int = 720,
                 motion_clip_ms: int = 2000, motion_clip_fps: int = 15):
        camera_type = (camera_type or "dslr").strip().lower()
        if camera_type not in CAMERA_TYPES:
            raise ValueError(f"Invalid camera type '{camera_type}'. Expected one of: {', '.join(CAMERA_TYPES)}")
        self._camera_type = camera_type
        self._live_view_fps = live_view_fps
        # Active face/color filter, applied to the live view and baked into photos
        self._filters = filters

        # Animated capture settings (see capture_clip)
        self._gif_frames = gif_frames
        self._gif_interval_ms = gif_interval_ms
        self._boomerang_duration_ms = boomerang_duration_ms
        self._boomerang_fps = boomerang_fps
        self._clip_gif_width = clip_gif_width
        # Still photos also get a motion clip of the live view leading up to the shutter (0 disables)
        self._motion_clip_ms = motion_clip_ms
        self._motion_clip_fps = motion_clip_fps

        if self.is_webcam:
            self._video_device = _parse_device(webcam_device)
        else:
            # We try to ensure no background process is holding the camera interface on Linux
            self._kill_gvfs_processes()
            self._video_device = _parse_device(live_view_device)

        self._latest_frame = None
        # Handle of the active live view stream, shared with webcam captures
        self._stream_cap = None
        self._capture_lock = asyncio.Lock()

        # A single producer task reads the device and fans encoded frames out to all viewers
        self._latest_jpeg: Optional[bytes] = None
        # Monotonic time the latest frame was read, used to space out clip frames
        self._latest_ts = 0.0
        self._frame_seq = 0
        self._frame_cond = asyncio.Condition()
        self._subscribers = 0
        self._producer_task: Optional[asyncio.Task] = None
        # Why the last producer stopped: "idle" (no viewers left) or "read_failed"
        self._producer_exit: Optional[str] = None
        # Set while a GIF/boomerang is recording; other captures are refused meanwhile
        self._recording = False
        # Rolling (timestamp, jpeg) window of the last motion_clip_ms of live view
        self._motion_buffer: collections.deque[tuple[float, bytes]] = collections.deque()

    @property
    def is_webcam(self) -> bool:
        return self._camera_type == "webcam"

    def _open_capture(self, resolution: tuple, fps: Optional[int] = None) -> cv2.VideoCapture:
        """
        Opens the configured video device (capture card or webcam) at the requested
        (width, height) and, optionally, frame rate. Blocking; run in a thread.
        """
        if sys.platform.startswith("linux"):
            cap = cv2.VideoCapture(self._video_device, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self._video_device)
        if self.is_webcam or (fps and fps > 30):
            # Most USB devices only deliver high resolutions / frame rates in MJPG
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._set_resolution(cap, resolution, fps)
        return cap

    @staticmethod
    def _set_resolution(cap: cv2.VideoCapture, resolution: tuple, fps: Optional[int] = None):
        """
        Requests a (width, height) and, optionally, a frame rate on an open capture.
        With V4L2, OpenCV restarts streaming internally when the size changes, which
        resets the frame rate, so fps is set after the size. Blocking; run in a thread.
        """
        width, height = resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        if not cap.isOpened():
            return
        actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual != (width, height):
            print(f"Warning: requested {width}x{height} but device is using {actual[0]}x{actual[1]}")
        actual_fps = round(cap.get(cv2.CAP_PROP_FPS))
        if fps and actual_fps != fps:
            print(f"Warning: requested {fps}fps but device is using {actual_fps}fps")

    def _check_webcam_connection(self) -> dict:
        """
        Checks that the webcam device node exists without opening it,
        so it doesn't conflict with an active stream.
        """
        if isinstance(self._video_device, int):
            node = f"/dev/video{self._video_device}"
        else:
            node = os.path.realpath(self._video_device)

        if not os.path.exists(node):
            return {"connected": False, "raw_output": f"{node} not found"}

        model = "USB Webcam"
        try:
            with open(f"/sys/class/video4linux/{os.path.basename(node)}/name") as f:
                model = f.read().strip() or model
        except OSError:
            pass

        return {"connected": True, "model": model, "raw_output": node}

    def _kill_gvfs_processes(self):
        """
        On Linux systems (like Ubuntu/Raspberry Pi), gvfs-gphoto2-volume-monitor
        can automatically grab the camera, preventing gphoto2 from accessing it.
        We attempt to kill these processes.
        """
        try:
            os.system("pkill -f gvfs-gphoto2-volume-monitor")
            os.system("pkill -f gvfsd-gphoto2")
        except Exception:
            pass # Just a best-effort, might fail if not Linux or no permissions

    async def _run_command(self, *args, timeout: int = 30) -> tuple[int, str, str]:
        """
        Runs a gphoto2 command asynchronously.
        Returns a tuple of (return_code, stdout, stderr)
        """
        cmd = ["gphoto2"] + list(args)
        process = None
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            
            return process.returncode, stdout.decode().strip(), stderr.decode().strip()
        except asyncio.TimeoutError:
            if process:
                process.kill()
            raise CameraError(f"Command {' '.join(cmd)} timed out after {timeout} seconds")
        except Exception as e:
            raise CameraError(f"Error executing gphoto2: {str(e)}")

    async def check_connection(self) -> dict:
        """
        Checks if any camera is connected using gphoto2 --auto-detect,
        or checks the configured webcam device in webcam mode.
        """
        if self.is_webcam:
            return self._check_webcam_connection()

        return_code, stdout, stderr = await self._run_command("--auto-detect")
        
        # Example output of gphoto2 --auto-detect:
        # Model                          Port                                            
        # ----------------------------------------------------------
        # Canon EOS 5D Mark IV           usb:001,006     
        
        lines = stdout.split("\n")
        if len(lines) > 2:
            # First two lines are headers. If there's a third line, a camera is connected.
            camera_info = lines[2].strip()
            # Try to parse model and port roughly
            parts = camera_info.rsplit("usb:", 1)
            model = parts[0].strip() if len(parts) == 2 else camera_info
            
            return {
                "connected": True,
                "model": model,
                "raw_output": stdout
            }
        
        return {
            "connected": False,
            "raw_output": stdout
        }

    async def capture_photo(self, output_dir: str, file_prefix: str, event_callback: Optional[callable] = None) -> str:
        """
        Triggers the camera to capture an image and download it to output_dir.
        Returns the filename of the downloaded image.
        In webcam mode, the photo is taken from the webcam instead.
        """
        self._check_not_recording()

        if self.is_webcam:
            if event_callback:
                await event_callback({"type": "camera_action", "action": "capture_initiated"})
            try:
                filename = await self.capture_from_stream(output_dir, file_prefix, event_callback)
            except CameraError:
                if event_callback:
                    await event_callback({"type": "camera_action", "action": "capture_failed"})
                raise
            if event_callback:
                await event_callback({"type": "camera_action", "action": "capture_completed", "filename": filename})
            return filename

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if event_callback:
            await event_callback({"type": "camera_action", "action": "capture_initiated"})
        
        # We use a timestamp-based filename format to avoid conflicts
        filename = f"{_timestamped_name(file_prefix)}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # --capture-image-and-download takes the picture and pulls it from the camera RAM/SD
        # --force-overwrite avoids interactive prompts if the file exists
        # --filename dictates where to save it
        
        if event_callback:
            await event_callback({"type": "camera_action", "action": "capturing_image"})
            
        return_code, stdout, stderr = await self._run_command(
            "--capture-image-and-download",
            "--force-overwrite",
            "--filename", filepath,
            timeout=15 # Capturing can take a few seconds
        )
        
        if return_code != 0:
            if event_callback:
                await event_callback({"type": "camera_action", "action": "capture_failed"})
            raise CameraError(f"Failed to capture photo: {stderr or stdout}")
            
        # Verify the file was actually created
        if not os.path.exists(filepath):
            if event_callback:
                await event_callback({"type": "camera_action", "action": "capture_failed"})
            raise CameraError("Photo capture command succeeded, but file was not saved.")

        if self._filters and self._filters.is_active:
            await asyncio.to_thread(self._filter_photo_file, filepath)

        if event_callback:
            await event_callback({"type": "camera_action", "action": "capture_completed", "filename": filename})
            
        return filename

    async def _produce_frames(self):
        """
        Reads frames from the capture device and publishes them to all live view
        subscribers. Runs while at least one subscriber is connected.
        """
        # Open the capture device
        # We run this in a thread to not block the event loop
        cap = await asyncio.to_thread(self._open_capture, LIVE_VIEW_RESOLUTION, self._live_view_fps)

        if not cap.isOpened():
            await asyncio.to_thread(cap.release)
            raise CameraError(f"Cannot open capture device {self._video_device}")

        self._stream_cap = cap
        self._producer_exit = "idle"
        try:
            while self._subscribers > 0:
                # Read frame without blocking the event loop. The lock lets a webcam
                # capture take over the device between frames.
                async with self._capture_lock:
                    success, frame = await asyncio.to_thread(cap.read)
                read_ts = time.monotonic()
                if not success:
                    self._producer_exit = "read_failed"
                    break

                # Keep the raw frame so captures can apply the filter themselves
                self._latest_frame = frame.copy()

                # Filter and encode the frame once, shared by every subscriber
                ret, buffer = await asyncio.to_thread(self._render_live_frame, frame)
                if ret:
                    async with self._frame_cond:
                        self._latest_jpeg = buffer.tobytes()
                        self._latest_ts = read_ts
                        self._frame_seq += 1
                        self._frame_cond.notify_all()
                    self._buffer_motion_frame(read_ts, self._latest_jpeg)

                # Yield to the event loop; cap.read() already paces the loop to the device frame rate
                await asyncio.sleep(0)
        finally:
            self._latest_frame = None
            self._latest_jpeg = None
            self._motion_buffer.clear()
            self._stream_cap = None
            await asyncio.to_thread(cap.release)

    def _buffer_motion_frame(self, ts: float, jpeg: bytes):
        if self._motion_clip_ms <= 0:
            return
        self._motion_buffer.append((ts, jpeg))
        window = self._motion_clip_ms / 1000
        while ts - self._motion_buffer[0][0] > window:
            self._motion_buffer.popleft()

    def _render_live_frame(self, frame):
        if self._filters:
            frame = self._filters.apply_video(frame)
        return cv2.imencode('.jpg', frame)

    def _save_photo(self, filepath: str, frame) -> bool:
        """
        Applies the active filter to a captured frame and writes it. Blocking; run in a thread.
        """
        if self._filters:
            frame = self._filters.apply_image(frame)
        return cv2.imwrite(filepath, frame)

    def _filter_photo_file(self, filepath: str):
        """
        Applies the active filter to a photo downloaded from the DSLR, in place. Blocking; run in a thread.
        """
        frame = cv2.imread(filepath)
        if frame is None:
            print(f"Warning: could not read {filepath} to apply filter")
            return
        if not cv2.imwrite(filepath, self._filters.apply_image(frame), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            print(f"Warning: could not save filtered photo {filepath}")

    def _ensure_producer(self):
        if self._producer_task is None or self._producer_task.done():
            self._producer_task = asyncio.create_task(self._produce_frames())
            # Wake subscribers when the producer stops, whatever the reason
            self._producer_task.add_done_callback(lambda _: asyncio.create_task(self._notify_subscribers()))

    async def _notify_subscribers(self):
        async with self._frame_cond:
            self._frame_cond.notify_all()

    async def stop_live_view(self):
        """
        Stops the shared producer and releases the capture device.
        """
        task = self._producer_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, CameraError):
                pass

    async def _subscribe_frames(self):
        """
        Yields (timestamp, jpeg_bytes) for every new filtered live view frame. The device is
        opened once for all subscribers and released when the last one leaves. Returns when
        the producer stops for good (device read failure or shutdown).
        """
        self._subscribers += 1
        try:
            self._ensure_producer()
            last_seq = self._frame_seq
            while True:
                task = self._producer_task
                async with self._frame_cond:
                    await self._frame_cond.wait_for(
                        lambda: (self._frame_seq != last_seq and self._latest_jpeg is not None) or task.done()
                    )
                    frame = (self._latest_ts, self._latest_jpeg) if self._frame_seq != last_seq else None
                    last_seq = self._frame_seq

                if frame is not None:
                    yield frame
                elif task.done():
                    if not task.cancelled() and task.exception() is not None:
                        raise task.exception()
                    if task.cancelled() or self._producer_exit == "read_failed":
                        return
                    # The producer stopped for lack of viewers just as we joined; restart it
                    self._ensure_producer()
        finally:
            self._subscribers -= 1

    async def stream_live_view(self):
        """
        Stream live view from a USB capture device (e.g., HDMI capture card)
        or USB webcam using OpenCV. Any number of clients can subscribe; the
        device is opened once and released when the last client disconnects.
        """
        async with aclosing(self._subscribe_frames()) as frames:
            async for _, frame_bytes in frames:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n"
                )

    def _check_not_recording(self):
        if self._recording:
            raise CameraError("Camera busy recording a clip.")

    async def capture_clip(self, output_dir: str, file_prefix: str, kind: str, event_callback: Optional[callable] = None) -> dict:
        """
        Records an animated clip from the live view and saves it as a looping GIF and an MP4.
        "gif" is a stop-motion of a few evenly spaced poses; "boomerang" is a short burst
        played forward then backward. Frames are the live view's, so the active filter is
        already applied and the live view keeps running while recording.
        Returns {"gif": filename, "mp4": filename}.
        """
        if kind not in CLIP_KINDS:
            raise ValueError(f"Invalid clip kind '{kind}'. Expected one of: {', '.join(CLIP_KINDS)}")
        # Fail fast rather than queue: a late clip would record the wrong moment
        self._check_not_recording()
        if self._capture_lock.locked():
            raise CameraError("Camera busy taking a photo.")

        async def emit(action: str, **extra):
            if event_callback:
                await event_callback({"type": "camera_action", "action": action, **extra})

        await emit("capture_initiated")
        os.makedirs(output_dir, exist_ok=True)
        stem = _timestamped_name(file_prefix)
        gif_name, mp4_name = f"{stem}.gif", f"{stem}.mp4"

        try:
            self._recording = True
            try:
                if kind == "gif":
                    frames = await self._record_gif(emit)
                    fps = 1000 / self._gif_interval_ms
                else:
                    frames = await self._record_boomerang(emit)
                    fps = self._boomerang_fps
            finally:
                self._recording = False
            await emit("recording_completed")

            await emit("encoding")
            try:
                await asyncio.gather(
                    clip_encoder.encode_gif(frames, fps, self._clip_gif_width, os.path.join(output_dir, gif_name)),
                    clip_encoder.encode_mp4(frames, fps, os.path.join(output_dir, mp4_name)),
                )
            except clip_encoder.ClipEncodeError as e:
                raise CameraError(f"Failed to encode {kind}: {e}")
        except CameraError:
            await emit("capture_failed")
            raise

        await emit("capture_completed", filename=gif_name, video_filename=mp4_name)
        return {"gif": gif_name, "mp4": mp4_name}

    async def _record_gif(self, emit) -> list[bytes]:
        """
        Picks one frame every gif_interval_ms, starting with the first frame received.
        """
        interval = self._gif_interval_ms / 1000
        picked: list[bytes] = []
        start = 0.0

        async def on_frame(ts: float, jpeg: bytes) -> bool:
            nonlocal start
            if not picked:
                start = ts
                await emit("recording_started", total=self._gif_frames)
            if ts < start + len(picked) * interval:
                return False
            picked.append(jpeg)
            await emit("frame_captured", index=len(picked), total=self._gif_frames)
            return len(picked) == self._gif_frames

        await self._record(on_frame, self._gif_frames * interval)
        return picked

    async def _record_boomerang(self, emit) -> list[bytes]:
        """
        Records every frame for boomerang_duration_ms, resamples them to a steady
        boomerang_fps, then appends the frames in reverse so the clip loops seamlessly.
        """
        duration = self._boomerang_duration_ms / 1000
        recorded: list[tuple[float, bytes]] = []

        async def on_frame(ts: float, jpeg: bytes) -> bool:
            if not recorded:
                await emit("recording_started")
            elif ts - recorded[0][0] >= duration:
                return True
            recorded.append((ts, jpeg))
            return False

        await self._record(on_frame, duration)

        forward = _resample(recorded, duration, self._boomerang_fps)
        # Drop both end frames from the reversed half so the turnarounds don't stutter
        return forward + forward[-2:0:-1]

    def snapshot_motion(self) -> list[tuple[float, bytes]]:
        """
        Returns the live view frames of the last motion_clip_ms. Call it before taking the
        photo: the webcam's switch to photo resolution and the DSLR shutter stall the live view.
        Empty when the live view isn't running or motion clips are disabled.
        """
        return list(self._motion_buffer)

    async def save_motion_clip(self, frames: list[tuple[float, bytes]], output_dir: str, photo_filename: str) -> Optional[str]:
        """
        Encodes frames from snapshot_motion() as an MP4 named after the photo.
        Returns its filename, or None if there was nothing to save or encoding failed,
        since a missing clip must never fail the photo itself.
        """
        if len(frames) < 2:
            return None
        duration = frames[-1][0] - frames[0][0]
        mp4_name = f"{os.path.splitext(photo_filename)[0]}.mp4"
        try:
            await clip_encoder.encode_mp4(
                _resample(frames, duration, self._motion_clip_fps),
                self._motion_clip_fps,
                os.path.join(output_dir, mp4_name),
            )
        except clip_encoder.ClipEncodeError as e:
            print(f"Warning: could not save motion clip for {photo_filename}: {e}")
            return None
        return mp4_name

    async def _record(self, on_frame, length: float):
        """
        Feeds live view frames to on_frame(ts, jpeg) until it returns True. Starts the
        device if no live view is running, skipping its first frames while exposure settles.
        """
        skip = 0 if self._stream_cap is not None else WARMUP_FRAMES
        try:
            async with asyncio.timeout(length + CLIP_FRAME_TIMEOUT):
                async with aclosing(self._subscribe_frames()) as frames:
                    async for ts, jpeg in frames:
                        if skip:
                            skip -= 1
                            continue
                        if await on_frame(ts, jpeg):
                            return
        except TimeoutError:
            raise CameraError("Timed out waiting for frames from capture device.")
        raise CameraError("Capture device stopped while recording.")

    async def capture_from_stream(self, output_dir: str, file_prefix: str, event_callback: Optional[callable] = None) -> str:
        """
        Captures a frame from the stream (either using the active stream's latest frame
        or opening the device temporarily) and saves it to output_dir.
        """
        self._check_not_recording()
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{_timestamped_name(file_prefix)}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # We use a lock to avoid multiple requests trying to read from the camera simultaneously
        async with self._capture_lock:
            stream_cap = self._stream_cap

            if self.is_webcam and stream_cap is not None:
                # Stream is active: briefly switch it to photo resolution, then back to HD
                def _grab_photo_from_stream():
                    try:
                        self._set_resolution(stream_cap, PHOTO_RESOLUTION)
                        # Discard the first frames after the mode switch while exposure settles
                        for _ in range(WARMUP_FRAMES):
                            stream_cap.read()
                        success, frame = stream_cap.read()
                    finally:
                        self._set_resolution(stream_cap, LIVE_VIEW_RESOLUTION, self._live_view_fps)
                    if not success:
                        raise CameraError("Failed to read frame from capture device.")
                    if not self._save_photo(filepath, frame):
                        raise CameraError("Failed to save frame from active stream.")
                await asyncio.to_thread(_grab_photo_from_stream)
                return filename

            if self._latest_frame is not None:
                # Stream is active, use a copy of the latest frame to prevent concurrent mutation
                frame_to_save = self._latest_frame.copy()
                success = await asyncio.to_thread(self._save_photo, filepath, frame_to_save)
                if not success:
                    raise CameraError("Failed to save frame from active stream.")
                return filename

            # Stream is not active, temporarily open the device
            resolution = PHOTO_RESOLUTION if self.is_webcam else LIVE_VIEW_RESOLUTION
            def _grab_frame():
                cap = self._open_capture(resolution)
                try:
                    if not cap.isOpened():
                        raise CameraError(f"Cannot open capture device {self._video_device}")
                    # Discard the first frames, which are often dark while exposure settles
                    for _ in range(WARMUP_FRAMES):
                        cap.read()
                    success, frame = cap.read()
                finally:
                    cap.release()
                if not success:
                    raise CameraError("Failed to read frame from capture device.")
                self._save_photo(filepath, frame)
            await asyncio.to_thread(_grab_frame)
            return filename
