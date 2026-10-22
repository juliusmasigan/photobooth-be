import asyncio
import shutil
from functools import lru_cache
from typing import List

# Encoding a few seconds of HD frames takes well under this, even on a Raspberry Pi
ENCODE_TIMEOUT = 60

class ClipEncodeError(Exception):
    pass

@lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    """
    Prefers a system ffmpeg; falls back to the static binary bundled with imageio-ffmpeg.
    """
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
    except ImportError:
        raise ClipEncodeError("ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.")
    return imageio_ffmpeg.get_ffmpeg_exe()

async def _run_ffmpeg(frames: List[bytes], fps: float, output_args: List[str]):
    """
    Pipes JPEG frames into ffmpeg at the given frame rate and runs it with output_args.
    """
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", f"{fps:g}", "-i", "-",
        *output_args,
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(b"".join(frames)), timeout=ENCODE_TIMEOUT)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise ClipEncodeError(f"ffmpeg timed out after {ENCODE_TIMEOUT} seconds")
    if process.returncode != 0:
        raise ClipEncodeError(stderr.decode().strip() or f"ffmpeg exited with code {process.returncode}")

async def encode_gif(frames: List[bytes], fps: float, width: int, path: str):
    """
    Writes a looping GIF no wider than width, using a palette built from this clip's own
    colors (much less banding than a generic 256-color palette).
    """
    await _run_ffmpeg(frames, fps, [
        "-filter_complex",
        f"[0:v]scale='min(iw,{width})':-1:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
        "-loop", "0",
        path,
    ])

async def encode_mp4(frames: List[bytes], fps: float, path: str):
    """
    Writes a browser-playable H.264 MP4. Stop-motion clips are output at 30fps (repeating
    frames) since some players handle very low frame rates poorly.
    """
    await _run_ffmpeg(frames, fps, [
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        *(["-r", "30"] if fps < 10 else []),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-movflags", "+faststart",
        path,
    ])
