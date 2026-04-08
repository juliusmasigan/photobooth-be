import asyncio
import os
import time
from typing import Optional
import cv2

class CameraError(Exception):
    pass

class CameraService:
    def __init__(self, live_view_usb_device_id: int = 0):
        # We try to ensure no background process is holding the camera interface on Linux
        self._kill_gvfs_processes()
        self._latest_frame = None
        self._capture_lock = asyncio.Lock()
        self._live_view_usb_device_id = int(live_view_usb_device_id)

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
        Checks if any camera is connected using gphoto2 --auto-detect.
        """
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

    async def capture_photo(self, output_dir: str, event_callback: Optional[callable] = None) -> str:
        """
        Triggers the camera to capture an image and download it to output_dir.
        Returns the filename of the downloaded image.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if event_callback:
            await event_callback({"type": "camera_action", "action": "capture_initiated"})
        
        # We use a timestamp-based filename format to avoid conflicts
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        filename = f"photo_{timestamp}_{ms:03d}.jpg"
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
        using OpenCV.
        """
        # Open the capture device
        # We run this in a thread to not block the event loop
        device_id: int = self._live_view_usb_device_id
        cap = await asyncio.to_thread(cv2.VideoCapture, device_id)
        
        if not cap.isOpened():
            raise CameraError(f"Cannot open capture device {device_id}")
            
        try:
            while True:
                # Read frame without blocking the event loop
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
            await asyncio.to_thread(cap.release)

    async def capture_from_stream(self, output_dir: str, event_callback: Optional[callable] = None) -> str:
        """
        Captures a frame from the stream (either using the active stream's latest frame
        or opening the device temporarily) and saves it to output_dir.
        """
        device_id: int = self._live_view_usb_device_id
        os.makedirs(output_dir, exist_ok=True)

        if event_callback:
            await event_callback({"type": "camera_action", "action": "capture_initiated"})

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        filename = f"photo_stream_{timestamp}_{ms:03d}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # We use a lock to avoid multiple requests trying to read from the camera simultaneously
        async with self._capture_lock:
            if self._latest_frame is not None:
                if event_callback:
                    await event_callback({"type": "camera_action", "action": "capturing_image"})

                # Stream is active, use a copy of the latest frame to prevent concurrent mutation
                frame_to_save = self._latest_frame.copy()
                success = await asyncio.to_thread(cv2.imwrite, filepath, frame_to_save)
                if not success:
                    if event_callback:
                        await event_callback({"type": "camera_action", "action": "capture_failed"})
                    raise CameraError("Failed to save frame from active stream.")
                if event_callback:
                    await event_callback({"type": "camera_action", "action": "capture_completed", "filename": filename})
                return filename
                
            # Stream is not active, temporarily open the device
            def _grab_frame():
                if event_callback:
                    asyncio.run(event_callback({"type": "camera_action", "action": "capturing_image"}))

                cap = cv2.VideoCapture(device_id)
                if not cap.isOpened():
                    if event_callback:
                        asyncio.run(event_callback({"type": "camera_action", "action": "capture_failed"}))
                    raise CameraError(f"Cannot open capture device {device_id}")
                success, frame = cap.read()
                cap.release()
                if not success:
                    if event_callback:
                        asyncio.run(event_callback({"type": "camera_action", "action": "capture_failed"}))
                    raise CameraError("Failed to read frame from capture device.")
                cv2.imwrite(filepath, frame)
                if event_callback:
                    asyncio.run(event_callback({"type": "camera_action", "action": "capture_completed", "filename": filename}))
            await asyncio.to_thread(_grab_frame)
            return filename
