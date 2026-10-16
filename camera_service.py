import asyncio
import os
import sys
import time
from typing import Optional, Union
import cv2

CAMERA_TYPES = ("dslr", "webcam")

# Number of frames to discard after opening a device, so auto-exposure/white balance can settle
WARMUP_FRAMES = 5

# Live view is kept at HD to stay light; webcam photos are taken at 4K
LIVE_VIEW_RESOLUTION = (1920, 1080)
PHOTO_RESOLUTION = (3840, 2160)

class CameraError(Exception):
    pass

def _parse_device(value) -> Union[int, str]:
    """
    Accepts a device index (e.g. 0, "2") or a device path
    (e.g. "/dev/video0", "/dev/v4l/by-id/usb-...-video-index0").
    """
    if value is None or str(value).strip() == "":
        return 0
    value = str(value).strip()
    return int(value) if value.isdigit() else value

class CameraService:
    def __init__(self, camera_type: str = "dslr", live_view_device=0, webcam_device=None):
        camera_type = (camera_type or "dslr").strip().lower()
        if camera_type not in CAMERA_TYPES:
            raise ValueError(f"Invalid camera type '{camera_type}'. Expected one of: {', '.join(CAMERA_TYPES)}")
        self._camera_type = camera_type

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

    @property
    def is_webcam(self) -> bool:
        return self._camera_type == "webcam"

    def _open_capture(self, resolution: tuple) -> cv2.VideoCapture:
        """
        Opens the configured video device (capture card or webcam) at the requested
        (width, height). Blocking; run in a thread.
        """
        if sys.platform.startswith("linux"):
            cap = cv2.VideoCapture(self._video_device, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self._video_device)
        if self.is_webcam:
            # Most USB webcams only deliver high resolutions at a usable frame rate in MJPG
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._set_resolution(cap, resolution)
        return cap

    @staticmethod
    def _set_resolution(cap: cv2.VideoCapture, resolution: tuple):
        """
        Requests a (width, height) on an open capture. With V4L2, OpenCV restarts
        streaming internally when the size changes. Blocking; run in a thread.
        """
        width, height = resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if cap.isOpened() and actual != (width, height):
            print(f"Warning: requested {width}x{height} but device is using {actual[0]}x{actual[1]}")

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
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        filename = f"{file_prefix}_{timestamp}_{ms:03d}.jpg"
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
            
        if event_callback:
            await event_callback({"type": "camera_action", "action": "capture_completed", "filename": filename})
            
        return filename

    async def stream_live_view(self):
        """
        Stream live view from a USB capture device (e.g., HDMI capture card)
        or USB webcam using OpenCV.
        """
        # Open the capture device
        # We run this in a thread to not block the event loop
        cap = await asyncio.to_thread(self._open_capture, LIVE_VIEW_RESOLUTION)

        if not cap.isOpened():
            await asyncio.to_thread(cap.release)
            raise CameraError(f"Cannot open capture device {self._video_device}")

        self._stream_cap = cap
        try:
            while True:
                # Read frame without blocking the event loop. The lock lets a webcam
                # capture take over the device between frames.
                async with self._capture_lock:
                    success, frame = await asyncio.to_thread(cap.read)
                if not success:
                    break
                    
                self._latest_frame = frame.copy()
                
                # Encode the frame as JPEG
                ret, buffer = await asyncio.to_thread(cv2.imencode, '.jpg', frame)
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n"
                )
                
                # Small sleep to ensure we yield control to the event loop
                await asyncio.sleep(0.01)
        finally:
            self._latest_frame = None
            self._stream_cap = None
            await asyncio.to_thread(cap.release)

    async def capture_from_stream(self, output_dir: str, file_prefix: str, event_callback: Optional[callable] = None) -> str:
        """
        Captures a frame from the stream (either using the active stream's latest frame
        or opening the device temporarily) and saves it to output_dir.
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        filename = f"{file_prefix}_{timestamp}_{ms:03d}.jpg"
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
                        self._set_resolution(stream_cap, LIVE_VIEW_RESOLUTION)
                    if not success:
                        raise CameraError("Failed to read frame from capture device.")
                    if not cv2.imwrite(filepath, frame):
                        raise CameraError("Failed to save frame from active stream.")
                await asyncio.to_thread(_grab_photo_from_stream)
                return filename

            if self._latest_frame is not None:
                # Stream is active, use a copy of the latest frame to prevent concurrent mutation
                frame_to_save = self._latest_frame.copy()
                success = await asyncio.to_thread(cv2.imwrite, filepath, frame_to_save)
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
                cv2.imwrite(filepath, frame)
            await asyncio.to_thread(_grab_frame)
            return filename
