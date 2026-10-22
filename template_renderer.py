import asyncio
import base64
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import cairosvg
import cv2
import numpy as np
from PIL import Image, ImageSequence

import clip_encoder

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

# Keep the template's prefixes when the edited tree is written back out
for _prefix, _uri in (
    ("", SVG_NS),
    ("xlink", XLINK_NS),
    ("inkscape", "http://www.inkscape.org/namespaces/inkscape"),
    ("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"),
):
    ET.register_namespace(_prefix, _uri)

# Photos are downscaled to this long edge before embedding; plenty for a slot on a 300 DPI print
MAX_PHOTO_EDGE = 3000
PHOTO_JPEG_QUALITY = 95
# Frame duration assumed for a GIF that doesn't state one
DEFAULT_FRAME_MS = 100
# Inches per unit for SVG lengths; unitless and px lengths are CSS pixels (96 per inch)
UNIT_INCHES = {"in": 1, "cm": 1 / 2.54, "mm": 1 / 25.4, "pt": 1 / 72, "pc": 1 / 6, "px": 1 / 96, "": 1 / 96}

def _read_photo(path: str) -> np.ndarray:
    """
    Loads a capture as BGR; GIF clips (which imread can't decode) use their first frame.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        cap = cv2.VideoCapture(path)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read photo {os.path.basename(path)}")
    return img

def _read_clip(path: str) -> Tuple[List[np.ndarray], int]:
    """
    Loads every frame of a GIF clip as BGR, with its frame duration in ms.
    Any other photo is a single frame.
    """
    if not path.lower().endswith(".gif"):
        return [_read_photo(path)], DEFAULT_FRAME_MS
    try:
        with Image.open(path) as gif:
            frames = [cv2.cvtColor(np.asarray(f.convert("RGB")), cv2.COLOR_RGB2BGR) for f in ImageSequence.Iterator(gif)]
            duration = gif.info.get("duration") or DEFAULT_FRAME_MS
    except OSError as e:
        raise RuntimeError(f"Could not read clip {os.path.basename(path)}: {e}")
    return frames, duration

def _encode_jpeg(img: np.ndarray, name: str, quality: int = PHOTO_JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode {name}")
    return buf.tobytes()

def _jpeg_data_uri(img: np.ndarray, name: str) -> str:
    """
    Returns the image as a JPEG data URI, downscaled to MAX_PHOTO_EDGE.
    """
    h, w = img.shape[:2]
    scale = MAX_PHOTO_EDGE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(_encode_jpeg(img, name)).decode("ascii")

def _length_inches(value: str | None) -> float | None:
    match = re.fullmatch(r"\s*([0-9.]+)\s*([a-z]*)\s*", value or "")
    if not match or match.group(2) not in UNIT_INCHES:
        return None
    return float(match.group(1)) * UNIT_INCHES[match.group(2)]

def _print_size(root: ET.Element, dpi: int) -> Tuple[int, int]:
    """
    Pixel size of the template printed at dpi, from its width/height (or viewBox, as CSS px).
    """
    width = _length_inches(root.get("width"))
    height = _length_inches(root.get("height"))
    if width is None or height is None:
        view_box = (root.get("viewBox") or "").replace(",", " ").split()
        if len(view_box) != 4:
            raise ValueError("Template has no usable width/height or viewBox")
        width, height = float(view_box[2]) / 96, float(view_box[3]) / 96
    return round(width * dpi), round(height * dpi)

def _slot_elements(root: ET.Element, slot_ids, svg_path: str) -> Dict[str, ET.Element]:
    """
    Maps each slot id to the template's <image> placeholder of the same id.
    Raises ValueError if a placeholder is missing from the template.
    """
    images = {el.get("id"): el for el in root.iter(f"{{{SVG_NS}}}image") if el.get("id")}
    missing = [slot_id for slot_id in slot_ids if slot_id not in images]
    if missing:
        raise ValueError(f'Placeholder "{missing[0]}" not found in {os.path.basename(svg_path)}')
    return {slot_id: images[slot_id] for slot_id in slot_ids}

def _set_slot(el: ET.Element, data_uri: str):
    """
    Puts the image into a placeholder, cropped to fill like the frontend preview.
    """
    el.set(f"{{{XLINK_NS}}}href", data_uri)
    el.set("href", data_uri)
    el.set("preserveAspectRatio", "xMidYMid slice")

def _rasterize(root: ET.Element, width: int, height: int) -> np.ndarray:
    """
    Renders the template to a BGR image, with transparent areas flattened onto white paper.
    """
    png = cairosvg.svg2png(bytestring=ET.tostring(root), output_width=width, output_height=height)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3].astype(np.float32) * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return img

def render_template(svg_path: str, slots: Dict[str, str], out_path: str, dpi: int = 300, quality: int = 95):
    """
    Puts each photo into the template's <image> placeholder of the same id (slots maps id -> photo path),
    cropped to fill like the frontend preview, and writes the result as a JPEG printed at dpi.
    Raises ValueError if a placeholder is missing from the template.
    """
    root = ET.parse(svg_path).getroot()
    for slot_id, el in _slot_elements(root, slots, svg_path).items():
        photo_path = slots[slot_id]
        _set_slot(el, _jpeg_data_uri(_read_photo(photo_path), os.path.basename(photo_path)))

    width, height = _print_size(root, dpi)
    jpeg = _encode_jpeg(_rasterize(root, width, height), os.path.basename(out_path), quality)
    # Written under a temporary name and swapped in, so a crash never leaves a partial print behind
    tmp_path = f"{out_path}.part"
    with open(tmp_path, "wb") as f:
        f.write(jpeg)
    os.replace(tmp_path, out_path)

def _png_data_uri(img: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode("ascii")

def _rasterize_rgba(root: ET.Element, width: int, height: int) -> np.ndarray:
    png = cairosvg.svg2png(bytestring=ET.tostring(root), output_width=width, output_height=height)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    return img if img.ndim == 3 and img.shape[2] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

def _slots_only(root: ET.Element, slots: Dict[str, ET.Element]) -> ET.Element:
    """
    Strips the template down to its slots, their ancestors (for transforms and clips) and <defs>.
    """
    keep = set(slots.values())
    parents = {child: parent for parent in root.iter() for child in parent}
    for el in slots.values():
        while el in parents:
            el = parents[el]
            keep.add(el)

    def prune(el: ET.Element):
        for child in list(el):
            if child.tag == f"{{{SVG_NS}}}defs":
                continue
            if child in keep:
                prune(child)
            else:
                el.remove(child)
    prune(root)
    return root

def _slot_placement(slots_root: ET.Element, elements: Dict[str, ET.Element], slot_id: str,
                    photo_size: Tuple[int, int], width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measures where a photo of photo_size (w, h) lands in the rendered slot: renders a horizontal and
    a vertical gradient in it and fits the (always affine) output -> photo pixel mapping.
    Returns that 2x3 matrix and the slot's coverage (alpha, 0-1) in the output.
    """
    pw, ph = photo_size
    ramps = (np.tile(np.linspace(0, 255, pw), (ph, 1)), np.tile(np.linspace(0, 255, ph)[:, None], (1, pw)))
    hidden = _png_data_uri(np.zeros((1, 1, 4), np.uint8))
    for other_id, el in elements.items():
        if other_id != slot_id:
            _set_slot(el, hidden)

    renders = []
    for ramp in ramps:
        _set_slot(elements[slot_id], _png_data_uri(cv2.cvtColor(ramp.round().astype(np.uint8), cv2.COLOR_GRAY2BGR)))
        renders.append(_rasterize_rgba(slots_root, width, height).astype(np.float32))
    coverage = renders[0][:, :, 3] / 255

    # Fit only fully covered pixels away from the edges, where the gradient isn't blended
    inside = cv2.erode((coverage == 1).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    v, u = np.nonzero(inside)
    if len(u) < 3:
        raise ValueError(f'Placeholder "{slot_id}" is not visible in the template')
    points = np.column_stack([u, v, np.ones_like(u)]).astype(np.float64)
    rows = []
    for render, size in zip(renders, (pw, ph)):
        target = render[v, u, 0].astype(np.float64) * (size - 1) / 255
        rows.append(np.linalg.lstsq(points, target, rcond=None)[0])
    return np.array(rows, np.float32), coverage

def _render_frames(svg_path: str, slots: Dict[str, str], width: int) -> Tuple[List[bytes], float]:
    """
    Renders the template once per frame of its clips, width pixels wide, as JPEGs.
    Returns the frames and their frame rate (the longest clip's).

    Rasterizing the template (with its embedded artwork) is slow, so it is done only twice: with the
    slots filled black, then white. Their difference is how much of the photo shows through each pixel,
    so each frame is black + difference * photo, with every clip frame warped into its slot's position.
    """
    root = ET.parse(svg_path).getroot()
    elements = _slot_elements(root, slots, svg_path)
    clips = {slot_id: _read_clip(path) for slot_id, path in slots.items()}
    # Shorter clips and still photos loop alongside the longest clip
    frame_count, duration = max(((len(frames), ms) for frames, ms in clips.values()), default=(1, DEFAULT_FRAME_MS))

    print_w, print_h = _print_size(root, 96)
    height = round(width * print_h / print_w)
    layers = []
    for value in (0, 255):
        for el in elements.values():
            _set_slot(el, _png_data_uri(np.full((1, 1, 3), value, np.uint8)))
        layers.append(_rasterize(root, width, height).astype(np.float32))
    black, show_through = layers[0], (layers[1] - layers[0]) / 255

    slots_root = _slots_only(root, elements)
    placements = {
        slot_id: _slot_placement(slots_root, elements, slot_id, frames[0].shape[1::-1], width, height)
        for slot_id, (frames, _) in clips.items()
    }
    # Where slots overlap (only at anti-aliased edges, normally), each pixel takes the most covering slot
    owner = np.argmax(np.stack([coverage for _, coverage in placements.values()]), axis=0)

    name = os.path.basename(svg_path)
    rendered = []
    for i in range(frame_count):
        photo = np.zeros_like(black)
        for n, (slot_id, (frames, _)) in enumerate(clips.items()):
            matrix, _ = placements[slot_id]
            warped = cv2.warpAffine(frames[i % len(frames)], matrix, (width, height),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            photo[owner == n] = warped[owner == n]
        frame = np.clip(black + show_through * photo, 0, 255).astype(np.uint8)
        rendered.append(_encode_jpeg(frame, name))
    return rendered, 1000 / duration

async def render_animated_template(svg_path: str, slots: Dict[str, str], out_path: str, width: int):
    """
    Like render_template, but plays GIF clips inside their placeholders and writes a looping GIF
    width pixels wide. Raises ValueError if a placeholder is missing from the template.
    """
    frames, fps = await asyncio.to_thread(_render_frames, svg_path, slots, width)
    # Keeps a .gif extension so ffmpeg picks the GIF muxer; swapped in once complete
    tmp_path = f"{out_path}.part.gif"
    try:
        await clip_encoder.encode_gif(frames, fps, width, tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
