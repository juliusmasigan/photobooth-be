from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
import uvicorn
import asyncio
from typing import List
from dotenv import load_dotenv
from operator import itemgetter
from datetime import datetime, timedelta

from camera_service import CameraService, CameraError

load_dotenv()

# "dslr" (gphoto2 capture + HDMI capture card live view) or "webcam" (USB webcam for both)
CAMERA_TYPE = os.getenv("CAMERA_TYPE", "dslr")
# Device index (e.g. 2) or path (e.g. /dev/v4l/by-id/...) of the capture card used in dslr mode
LIVE_VIEW_USB_DEVICE_ID = os.getenv("LIVE_VIEW_USB_DEVICE_ID")
# Device index (e.g. 0) or path (e.g. /dev/v4l/by-id/...) of the webcam used in webcam mode
WEBCAM_DEVICE = os.getenv("WEBCAM_DEVICE")

# Connection Manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()
camera = CameraService(
    camera_type=CAMERA_TYPE,
    live_view_device=LIVE_VIEW_USB_DEVICE_ID,
    webcam_device=WEBCAM_DEVICE,
)

async def poll_camera_status():
    last_status = None
    while True:
        try:
            status_info = await camera.check_connection()
            current_status = status_info.get("connected", False)
            if last_status != current_status:
                last_status = current_status
                await manager.broadcast({
                    "type": "camera_status",
                    "connected": current_status,
                    "model": status_info.get("model")
                })
        except Exception:
            pass
        await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background polling task
    polling_task = asyncio.create_task(poll_camera_status())
    yield
    # Cleanup task on shutdown
    polling_task.cancel()

# Setup FastAPI App
app = FastAPI(
    title="DSLR Camera Controller API",
    description="API to control DSLR cameras via gphoto2, or USB webcams, for photobooth applications.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow CORS for front-end photobooth apps to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to specific frontend domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a photos directory to store the images
PHOTOS_DIR = os.getenv("PHOTOS_DIR", "photos")
os.makedirs(PHOTOS_DIR, exist_ok=True)

# Mount a static directory so we can access downloaded images via URL
app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")

class StatusResponse(BaseModel):
    connected: bool
    model: str | None = None
    raw_output: str

class CaptureRequest(BaseModel):
    id: str
    method: str = "gphoto"  # "gphoto" or "stream"
    mode: str | None = None  # Future use for different capture modes (e.g., burst, timer)

class CaptureResponse(BaseModel):
    success: bool
    filename: str | None = None
    url: str | None = None
    error: str | None = None

class PhotoResponse(BaseModel):
    id: str
    path: str
    time: str

class PhotoListResponse(BaseModel):
    photos: List[PhotoResponse]

@app.get("/", tags=["General"])
async def read_root():
    return {"message": "DSLR Camera Controller API is running. Check /docs for API documentation."}

@app.get("/api/photos", response_model=PhotoListResponse, tags=["Photos"])
async def list_photos(prefix: str = None):
    """
    List all photo URLs, newest first.
    """
    if not os.path.exists(PHOTOS_DIR):
        return PhotoListResponse(photos=[])

    try:
        # Photos are written once and never modified, so mtime is their creation time
        entries: List = []
        with os.scandir(PHOTOS_DIR) as it:
            for entry in it:
                captured_time = datetime.fromtimestamp(entry.stat().st_mtime)
                elapsed_time = datetime.now() - captured_time
                if entry.is_file() and not prefix:
                    entries.append({
                        'id': entry.name.split("_")[0],
                        'captured_time': captured_time,
                        'time': f"{int(elapsed_time.seconds // 3600)}h {int(elapsed_time.seconds % 60)}m ago",
                        'path': f"/photos/{entry.name}"
                    })
                elif entry.is_file() and entry.name.startswith(prefix):
                    entries.append({
                        'id': entry.name.split("_")[0],
                        'captured_time': captured_time,
                        'time': f"{int(elapsed_time.seconds // 3600)}h {int(elapsed_time.seconds % 60)}m ago",
                        'path': f"/photos/{entry.name}"
                    })
        entries.sort(key=itemgetter('captured_time'), reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading photos directory: {str(e)}")
        
    return PhotoListResponse(photos=entries)

@app.get("/api/status", response_model=StatusResponse, tags=["Camera"])
async def get_camera_status():
    """
    Check if the camera is connected and recognized by gphoto2.
    """
    try:
        status_info = await camera.check_connection()
        return StatusResponse(**status_info)
    except CameraError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/api/capture", response_model=CaptureResponse, tags=["Camera"])
async def capture_photo(request: CaptureRequest | None = None):
    """
    Trigger the camera to take a photo, download it to the server, and return the image URL.
    Optionally specify the method ("gphoto" or "stream").
    """
    try:
        capture_method = request.method if request else "gphoto"
        session_id = request.id if request else None
        
        await manager.broadcast({
            "type": "capture_started",
            "session_id": session_id,
            "method": capture_method
        })

        async def on_camera_event(event_data):
            event_data["session_id"] = session_id
            await manager.broadcast(event_data)
        
        if capture_method == "stream":
            filename = await camera.capture_from_stream(PHOTOS_DIR, session_id, event_callback=on_camera_event)
        else:
            filename = await camera.capture_photo(PHOTOS_DIR, session_id, event_callback=on_camera_event)
        
        # Construct the URL for the frontend to access the image
        # In a real deployed app, you might need to use request.base_url to form absolute URLs
        photo_url = f"/photos/{filename}"
        
        await manager.broadcast({
            "type": "capture_success",
            "session_id": session_id,
            "filename": filename,
            "url": photo_url
        })
        
        return CaptureResponse(
            success=True,
            filename=filename,
            url=photo_url
        )
    except CameraError as e:
        session_id = request.id if request else None
        await manager.broadcast({
            "type": "capture_error",
            "session_id": session_id,
            "error": str(e)
        })
        return CaptureResponse(success=False, error=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error during capture: {str(e)}")

@app.get("/api/stream", tags=["Camera"])
async def stream_camera():
    """
    Stream the live view from the camera using MJPEG format.
    Use this endpoint directly in an HTML <img> tag's src attribute.
    """
    return StreamingResponse(
        camera.stream_live_view(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We just need to keep the connection open and listen for pings
            data = await websocket.receive_text()
            # Respond to ping or other messages if necessary
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    # Start the server with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
