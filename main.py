from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
import json
import uvicorn
import asyncio
import hashlib
import re
from typing import List
from dotenv import load_dotenv
from operator import itemgetter
from datetime import datetime, timedelta

from camera_service import CameraService, CameraError, CLIP_KINDS
from filters import FilterEngine
from template_renderer import render_animated_template, render_template

load_dotenv()

# "dslr" (gphoto2 capture + HDMI capture card live view) or "webcam" (USB webcam for both)
CAMERA_TYPE = os.getenv("CAMERA_TYPE", "dslr")
# Device index (e.g. 2) or path (e.g. /dev/v4l/by-id/...) of the capture card used in dslr mode
LIVE_VIEW_USB_DEVICE_ID = os.getenv("LIVE_VIEW_USB_DEVICE_ID")
# Device index (e.g. 0) or path (e.g. /dev/v4l/by-id/...) of the webcam used in webcam mode
WEBCAM_DEVICE = os.getenv("WEBCAM_DEVICE")
# Live view frame rate requested from the capture device (e.g. 30 or 60)
LIVE_VIEW_FPS = int(os.getenv("LIVE_VIEW_FPS", "30"))
# Filter active at startup (see GET /api/filters for ids), e.g. "sunglasses"; defaults to none
DEFAULT_FILTER = os.getenv("DEFAULT_FILTER")
# Animated GIF (stop-motion): number of poses and the time between them
GIF_FRAMES = int(os.getenv("GIF_FRAMES", "6"))
GIF_INTERVAL_MS = int(os.getenv("GIF_INTERVAL_MS", "500"))
# Boomerang: length of the recorded burst (it plays forward then back) and its frame rate
BOOMERANG_DURATION_MS = int(os.getenv("BOOMERANG_DURATION_MS", "1000"))
BOOMERANG_FPS = int(os.getenv("BOOMERANG_FPS", "10"))
# Maximum width of GIF output; the MP4 keeps the live view resolution
CLIP_GIF_WIDTH = int(os.getenv("CLIP_GIF_WIDTH", "720"))
# Still photos also save an MP4 of the live view leading up to the shutter: its length (0 = off) and frame rate
MOTION_CLIP_MS = int(os.getenv("MOTION_CLIP_MS", "2000"))
MOTION_CLIP_FPS = int(os.getenv("MOTION_CLIP_FPS", "15"))

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

# Remote control relay: pairs one booth with up to MAX_REMOTES phones per session
MAX_REMOTES = 4
MAX_REMOTE_MESSAGE_BYTES = 4096
# Message types a remote (guest phone) is allowed to send to the booth
REMOTE_ALLOWED_TYPES = {"remote_command", "remote_hello"}

class RemoteRelay:
    def __init__(self):
        self.rooms: dict = {}

    def room(self, session_id: str) -> dict:
        return self.rooms.setdefault(session_id, {"booth": None, "remotes": set()})

    def cleanup(self, session_id: str):
        room = self.rooms.get(session_id)
        if room and room["booth"] is None and not room["remotes"]:
            del self.rooms[session_id]

    async def send_to_booth(self, room: dict, message: dict | str):
        booth = room["booth"]
        if booth is None:
            return
        try:
            if isinstance(message, str):
                await booth.send_text(message)
            else:
                await booth.send_json(message)
        except Exception:
            if room["booth"] is booth:
                room["booth"] = None

    async def send_to_remotes(self, room: dict, message: dict | str):
        for remote in list(room["remotes"]):
            try:
                if isinstance(message, str):
                    await remote.send_text(message)
                else:
                    await remote.send_json(message)
            except Exception:
                room["remotes"].discard(remote)

relay = RemoteRelay()

filter_engine = FilterEngine(default_filter=DEFAULT_FILTER)

camera = CameraService(
    camera_type=CAMERA_TYPE,
    live_view_device=LIVE_VIEW_USB_DEVICE_ID,
    webcam_device=WEBCAM_DEVICE,
    live_view_fps=LIVE_VIEW_FPS,
    filters=filter_engine,
    gif_frames=GIF_FRAMES,
    gif_interval_ms=GIF_INTERVAL_MS,
    boomerang_duration_ms=BOOMERANG_DURATION_MS,
    boomerang_fps=BOOMERANG_FPS,
    clip_gif_width=CLIP_GIF_WIDTH,
    motion_clip_ms=MOTION_CLIP_MS,
    motion_clip_fps=MOTION_CLIP_FPS,
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
    # Release the capture device if a live view is still running
    await camera.stop_live_view()
    filter_engine.close()

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
# Files listed by /api/photos as a capture's image; a same-named .mp4 is its video
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif")

# Mount a static directory so we can access downloaded images via URL
app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")

# Print templates: SVG designs plus layouts.json, which lists them by layout type
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "templates")
TEMPLATE_LAYOUTS_FILE = os.path.join(TEMPLATES_DIR, "layouts.json")
# Serves the template SVGs at the /designs/... urls used in layouts.json
app.mount("/designs", StaticFiles(directory=TEMPLATES_DIR), name="designs")

def load_template_layouts() -> dict:
    """
    Reads layouts.json on each call, so edits show up without a restart.
    """
    try:
        with open(TEMPLATE_LAYOUTS_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Could not load template layouts: {e}")

# Rendered template prints (JPEG at PRINT_DPI, or GIF for animated layouts) for printing and download
PRINTS_DIR = os.getenv("PRINTS_DIR", "prints")
os.makedirs(PRINTS_DIR, exist_ok=True)
PRINT_DPI = int(os.getenv("PRINT_DPI", "300"))
# Layouts whose clips play inside the template, rendered as a looping GIF CLIP_GIF_WIDTH wide
ANIMATED_LAYOUTS = ("gif", "boomerang")
PRINT_EXTENSIONS = (".jpg", ".gif")
# CUPS printer used by /api/print; unset uses the system default destination
PRINTER_NAME = os.getenv("PRINTER_NAME")
# Extra lp -o options, space separated, e.g. "media=4x6 fit-to-page"
PRINT_OPTIONS = os.getenv("PRINT_OPTIONS", "fit-to-page").split()
MAX_PRINT_COPIES = int(os.getenv("MAX_PRINT_COPIES", "5"))
# lp only queues the job, so it returns quickly unless CUPS is unreachable
PRINT_SUBMIT_TIMEOUT = 15

class DownloadStaticFiles(StaticFiles):
    """StaticFiles that asks browsers to save the file instead of displaying it."""
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Content-Disposition"] = f'attachment; filename="{os.path.basename(path)}"'
        return response

# QR code scans download the print rather than open it in the browser
app.mount("/prints", DownloadStaticFiles(directory=PRINTS_DIR), name="prints")

class StatusResponse(BaseModel):
    connected: bool
    model: str | None = None
    raw_output: str

class CaptureRequest(BaseModel):
    id: str
    method: str = "gphoto"  # "gphoto" or "stream"
    mode: str | None = None  # "gif" or "boomerang" records an animated clip; anything else takes a photo

class CaptureResponse(BaseModel):
    success: bool
    filename: str | None = None
    url: str | None = None
    video_url: str | None = None  # MP4 version of a gif/boomerang clip
    error: str | None = None

class PhotoResponse(BaseModel):
    id: str
    imageUrl: str
    videoUrl: str | None = None  # MP4 of a gif/boomerang clip, or a photo's motion clip
    time: str

class PhotoListResponse(BaseModel):
    photos: List[PhotoResponse]

class FilterInfo(BaseModel):
    id: str
    name: str
    kind: str  # "none", "sticker", "effect" or "color"

class FilterListResponse(BaseModel):
    active: str
    filters: List[FilterInfo]

class SetFilterRequest(BaseModel):
    id: str

class TemplateInfo(BaseModel):
    id: int
    name: str
    url: str
    imgIds: List[str]  # ids of the SVG elements that receive captured photos

class TemplateListResponse(BaseModel):
    templates: dict[str, List[TemplateInfo]]  # keyed by layout type, e.g. "collage", "still"

class RenderRequest(BaseModel):
    id: str | None = None  # session id, used as the file prefix like captures
    type: str  # layout type, e.g. "collage" or "still"
    templateId: int
    photos: List[str]  # selected photos' imageUrl (/photos/<file>) or filename, in imgIds order

class RenderResponse(BaseModel):
    success: bool
    filename: str
    url: str  # /prints/<file>.jpg, or .gif for animated layouts

class PrintInfo(BaseModel):
    id: str  # session id prefix of the file name ("print" when none was given)
    imageUrl: str  # /prints/<file>.jpg or .gif
    time: str

class PrintListResponse(BaseModel):
    prints: List[PrintInfo]

class PrintRequest(BaseModel):
    url: str  # url (/prints/<file>.jpg) or filename returned by /api/render
    copies: int = 1

class PrintResponse(BaseModel):
    success: bool
    job_id: str | None = None  # CUPS request id, e.g. "Booth_Printer-12"
    printer: str | None = None

@app.get("/", tags=["General"])
async def read_root():
    return {"message": "DSLR Camera Controller API is running. Check /docs for API documentation."}

def _age_label(mtime: float) -> str:
    """
    "<h>h <m>m ago" for a file written at mtime.
    """
    minutes = int((datetime.now() - datetime.fromtimestamp(mtime)).total_seconds() // 60)
    return f"{minutes // 60}h {minutes % 60}m ago"

@app.get("/api/photos", response_model=PhotoListResponse, tags=["Photos"])
async def list_photos(prefix: str = None):
    """
    List all photo URLs, newest first.
    """
    if not os.path.exists(PHOTOS_DIR):
        return PhotoListResponse(photos=[])

    try:
        # A capture's image and its MP4 share a file stem; group them into one entry
        images: dict = {}
        videos: dict = {}
        with os.scandir(PHOTOS_DIR) as it:
            for entry in it:
                if not entry.is_file() or (prefix and not entry.name.startswith(prefix)):
                    continue
                stem, ext = os.path.splitext(entry.name)
                ext = ext.lower()
                if ext in IMAGE_EXTENSIONS:
                    images[stem] = entry
                elif ext == ".mp4":
                    videos[stem] = entry.name

        entries: List = []
        for stem, image in images.items():
            # Photos are written once and never modified, so mtime is their creation time
            mtime = image.stat().st_mtime
            video = videos.get(stem)
            entries.append({
                'id': image.name.split("_")[0],
                'captured_time': datetime.fromtimestamp(mtime),
                'time': _age_label(mtime),
                'imageUrl': f"/photos/{image.name}",
                'videoUrl': f"/photos/{video}" if video else None
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
    With mode "gif" or "boomerang", records an animated clip from the live view instead:
    url is the looping GIF and video_url the same clip as MP4.
    For photos, video_url is a short MP4 of the live view just before the shutter, when it was running.
    """
    try:
        capture_method = request.method if request else "gphoto"
        session_id = request.id if request else None
        mode = request.mode if request else None

        await manager.broadcast({
            "type": "capture_started",
            "session_id": session_id,
            "method": capture_method,
            "mode": mode
        })

        async def on_camera_event(event_data):
            event_data["session_id"] = session_id
            await manager.broadcast(event_data)

        video_url = None
        if mode in CLIP_KINDS:
            clip = await camera.capture_clip(PHOTOS_DIR, session_id, mode, event_callback=on_camera_event)
            filename = clip["gif"]
            video_url = f"/photos/{clip['mp4']}"
        else:
            # Grab the moments before the shutter now; the live view stalls while the photo is taken
            motion_frames = camera.snapshot_motion()
            if capture_method == "stream":
                filename = await camera.capture_from_stream(PHOTOS_DIR, session_id, event_callback=on_camera_event)
            else:
                filename = await camera.capture_photo(PHOTOS_DIR, session_id, event_callback=on_camera_event)
            motion_name = await camera.save_motion_clip(motion_frames, PHOTOS_DIR, filename)
            if motion_name:
                video_url = f"/photos/{motion_name}"

        # Construct the URL for the frontend to access the image
        # In a real deployed app, you might need to use request.base_url to form absolute URLs
        photo_url = f"/photos/{filename}"

        await manager.broadcast({
            "type": "capture_success",
            "session_id": session_id,
            "filename": filename,
            "url": photo_url,
            "video_url": video_url
        })

        return CaptureResponse(
            success=True,
            filename=filename,
            url=photo_url,
            video_url=video_url
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

@app.get("/api/filters", response_model=FilterListResponse, tags=["Filters"])
async def list_filters():
    """
    List the available filters and the one currently active.
    """
    return FilterListResponse(active=filter_engine.active, filters=filter_engine.list())

@app.post("/api/filter", response_model=FilterListResponse, tags=["Filters"])
async def set_filter(request: SetFilterRequest):
    """
    Set the active filter for the live view and captured photos ("none" turns filters off).
    Connected /ws clients are notified with a filter_changed message.
    """
    try:
        filter_engine.set_active(request.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    await manager.broadcast({
        "type": "filter_changed",
        "filter": filter_engine.active
    })
    return FilterListResponse(active=filter_engine.active, filters=filter_engine.list())

@app.get("/api/templates", response_model=TemplateListResponse, tags=["Templates"])
async def list_templates():
    """
    List the print templates grouped by layout type, with each one's SVG url and photo slot ids.
    """
    return TemplateListResponse(templates=load_template_layouts())

# Renders still running, keyed by output file name
renders_in_progress: dict[str, asyncio.Task] = {}

@app.post("/api/render", response_model=RenderResponse, tags=["Templates"])
async def render_print(request: RenderRequest):
    """
    Render the selected photos into a template as a print-ready JPEG (PRINT_DPI) and return its URL.
    gif and boomerang layouts render as a looping GIF instead, with each clip playing in its slot.
    photos fill the template's imgIds slots in order, so there must be exactly one per slot.
    """
    layouts = load_template_layouts()
    template = next((t for t in layouts.get(request.type, []) if t.get("id") == request.templateId), None)
    if template is None:
        raise HTTPException(status_code=404, detail=f"Template {request.type}/{request.templateId} not found")

    img_ids = template.get("imgIds", [])
    if len(request.photos) != len(img_ids):
        raise HTTPException(status_code=400, detail=f"Template needs {len(img_ids)} photos, got {len(request.photos)}")

    photo_paths = []
    for photo in request.photos:
        # Only the file name is used, so a photo can't point outside PHOTOS_DIR
        name = os.path.basename(photo)
        path = os.path.join(PHOTOS_DIR, name)
        if not name.lower().endswith(IMAGE_EXTENSIONS) or not os.path.isfile(path):
            raise HTTPException(status_code=400, detail=f"Photo not found: {photo}")
        photo_paths.append(path)

    svg_path = os.path.join(TEMPLATES_DIR, template["url"].removeprefix("/designs/"))
    if not os.path.isfile(svg_path):
        raise HTTPException(status_code=404, detail=f"Template file not found: {template['url']}")
    # Same template + photos -> same file, so repeated requests (React StrictMode, retries, re-mounts) reuse one print
    animated = request.type in ANIMATED_LAYOUTS
    render_key = hashlib.sha1(json.dumps([
        request.type, request.templateId, [os.path.basename(p) for p in photo_paths],
        os.path.getmtime(svg_path), CLIP_GIF_WIDTH if animated else PRINT_DPI
    ]).encode()).hexdigest()[:12]
    prefix = os.path.basename(request.id or "") or "print"
    filename = f"{prefix}_{request.type}{request.templateId}_{render_key}{'.gif' if animated else '.jpg'}"
    out_path = os.path.join(PRINTS_DIR, filename)

    if not os.path.isfile(out_path):
        # A duplicate request that arrives mid-render waits on the first one instead of rendering again
        task = renders_in_progress.get(filename)
        if task is None:
            slots = dict(zip(img_ids, photo_paths))
            if animated:
                task = asyncio.create_task(render_animated_template(svg_path, slots, out_path, CLIP_GIF_WIDTH))
            else:
                task = asyncio.create_task(asyncio.to_thread(render_template, svg_path, slots, out_path, PRINT_DPI))
            renders_in_progress[filename] = task
            task.add_done_callback(lambda _: renders_in_progress.pop(filename, None))
        try:
            # Shielded so one client disconnecting doesn't cancel the render the others are waiting on
            await asyncio.shield(task)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error rendering print: {str(e)}")

    return RenderResponse(success=True, filename=filename, url=f"/prints/{filename}")

@app.get("/api/prints", response_model=PrintListResponse, tags=["Templates"])
async def list_prints(prefix: str = None):
    """
    List rendered prints, newest first. prefix filters by file name (e.g. a session id).
    """
    try:
        entries: List = []
        with os.scandir(PRINTS_DIR) as it:
            for entry in it:
                # .part files are renders still being written
                if not entry.is_file() or not entry.name.lower().endswith(PRINT_EXTENSIONS) or ".part" in entry.name:
                    continue
                if prefix and not entry.name.startswith(prefix):
                    continue
                # Prints are written once and never modified, so mtime is their creation time
                entries.append((entry.stat().st_mtime, entry.name))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Error reading prints directory: {str(e)}")

    entries.sort(reverse=True)
    return PrintListResponse(prints=[
        PrintInfo(id=name.split("_")[0], imageUrl=f"/prints/{name}", time=_age_label(mtime))
        for mtime, name in entries
    ])

@app.post("/api/print", response_model=PrintResponse, tags=["Templates"])
async def print_render(request: PrintRequest):
    """
    Send a print rendered by /api/render to the CUPS printer (PRINTER_NAME, or the system default).
    Connected /ws clients are notified with a print_queued message.
    """
    # Only the file name is used, so the url can't point outside PRINTS_DIR
    filename = os.path.basename(request.url)
    path = os.path.join(PRINTS_DIR, filename)
    if not filename or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Print not found: {request.url}")
    if not 1 <= request.copies <= MAX_PRINT_COPIES:
        raise HTTPException(status_code=400, detail=f"copies must be between 1 and {MAX_PRINT_COPIES}")

    cmd = ["lp", "-n", str(request.copies)]
    if PRINTER_NAME:
        cmd += ["-d", PRINTER_NAME]
    for option in PRINT_OPTIONS:
        cmd += ["-o", option]
    cmd.append(path)

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="lp not found; install CUPS to enable printing")
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), PRINT_SUBMIT_TIMEOUT)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise HTTPException(status_code=503, detail="Timed out submitting the print job to CUPS")
    if process.returncode != 0:
        raise HTTPException(status_code=503, detail=f"Print failed: {stderr.decode().strip()}")

    # lp prints "request id is <printer>-<n> (1 file(s))"
    match = re.search(r"request id is (\S+)", stdout.decode())
    job_id = match.group(1) if match else None
    printer = job_id.rsplit("-", 1)[0] if job_id else PRINTER_NAME

    await manager.broadcast({
        "type": "print_queued",
        "job_id": job_id,
        "printer": printer,
        "filename": filename
    })
    return PrintResponse(success=True, job_id=job_id, printer=printer)

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

@app.websocket("/ws/remote/{session_id}")
async def websocket_remote(websocket: WebSocket, session_id: str, role: str = "remote"):
    """
    Relay between a booth and the phones that scanned its QR code.
    Booth messages go to every remote; remote_command/remote_hello from a remote go to the booth.
    Presence: the booth gets remote_joined/remote_left {count}; remotes get booth_joined/booth_left.
    """
    if role not in ("booth", "remote"):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    room = relay.room(session_id)

    if role == "booth":
        stale_booth = room["booth"]
        room["booth"] = websocket
        if stale_booth is not None:
            try:
                await stale_booth.close()
            except Exception:
                pass
        await relay.send_to_remotes(room, {"type": "booth_joined"})
        if room["remotes"]:
            await relay.send_to_booth(room, {"type": "remote_joined", "count": len(room["remotes"])})
    else:
        if len(room["remotes"]) >= MAX_REMOTES:
            await websocket.close(code=1013)
            relay.cleanup(session_id)
            return
        room["remotes"].add(websocket)
        await relay.send_to_booth(room, {"type": "remote_joined", "count": len(room["remotes"])})
        if room["booth"] is not None:
            await websocket.send_json({"type": "booth_joined"})

    try:
        while True:
            data = await websocket.receive_text()
            if len(data) > MAX_REMOTE_MESSAGE_BYTES:
                continue
            try:
                message = json.loads(data)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue

            if role == "booth":
                await relay.send_to_remotes(room, data)
            elif message.get("type") in REMOTE_ALLOWED_TYPES:
                await relay.send_to_booth(room, data)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if role == "booth":
            # A replaced (stale) booth must not evict the booth that took its place
            if room["booth"] is websocket:
                room["booth"] = None
                await relay.send_to_remotes(room, {"type": "booth_left"})
        else:
            room["remotes"].discard(websocket)
            await relay.send_to_booth(room, {"type": "remote_left", "count": len(room["remotes"])})
        relay.cleanup(session_id)

if __name__ == "__main__":
    # Start the server with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
