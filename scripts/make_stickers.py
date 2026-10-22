"""
Generates the starter AR sticker PNGs in assets/filters/.
Shapes are drawn at 2x and downscaled with premultiplied alpha for clean, antialiased edges.

    python scripts/make_stickers.py
"""
import os

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "filters")
SS = 2  # supersampling factor

class Canvas:
    def __init__(self, width: int, height: int):
        self.size = (width, height)
        self.color = np.zeros((height * SS, width * SS, 3), dtype=np.uint8)
        self.alpha = np.zeros((height * SS, width * SS), dtype=np.uint8)

    @staticmethod
    def _pts(points):
        return [np.array(points, dtype=np.int32) * SS]

    def poly(self, points, color, alpha=255):
        cv2.fillPoly(self.color, self._pts(points), color, cv2.LINE_AA)
        cv2.fillPoly(self.alpha, self._pts(points), alpha, cv2.LINE_AA)

    def ellipse(self, center, axes, color, alpha=255, angle=0):
        c = (center[0] * SS, center[1] * SS)
        a = (axes[0] * SS, axes[1] * SS)
        cv2.ellipse(self.color, c, a, angle, 0, 360, color, -1, cv2.LINE_AA)
        cv2.ellipse(self.alpha, c, a, angle, 0, 360, alpha, -1, cv2.LINE_AA)

    def line(self, start, end, color, thickness, alpha=255):
        s = (start[0] * SS, start[1] * SS)
        e = (end[0] * SS, end[1] * SS)
        cv2.line(self.color, s, e, color, thickness * SS, cv2.LINE_AA)
        cv2.line(self.alpha, s, e, alpha, thickness * SS, cv2.LINE_AA)

    def save(self, name: str):
        alpha = self.alpha.astype(np.float32) / 255.0
        premultiplied = self.color.astype(np.float32) * alpha[:, :, None]
        premultiplied = cv2.resize(premultiplied, self.size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, self.size, interpolation=cv2.INTER_AREA)
        color = premultiplied / np.maximum(alpha[:, :, None], 1e-6)
        image = np.dstack([np.clip(color, 0, 255), alpha * 255]).astype(np.uint8)
        path = os.path.join(OUT_DIR, name)
        cv2.imwrite(path, image)
        print(f"Wrote {path}")

def sunglasses():
    # 600 px wide; at width 2.4 the lens centers (x = 175 and 425) sit about 1 eye distance apart
    c = Canvas(600, 240)
    frame = (25, 25, 25)
    for cx in (175, 425):
        c.ellipse((cx, 115), (122, 100), frame)
        c.ellipse((cx, 115), (108, 86), (60, 40, 30), alpha=235)
        c.ellipse((cx - 45, 75), (30, 13), (220, 210, 200), alpha=150, angle=-20)
    c.line((290, 90), (310, 90), frame, 16)
    c.line((55, 85), (5, 70), frame, 16)
    c.line((545, 85), (595, 70), frame, 16)
    c.save("sunglasses.png")

def mustache():
    c = Canvas(400, 140)
    brown = (25, 45, 75)
    # Two mirrored curls meeting under the nose
    left = [(200, 30), (160, 25), (110, 40), (70, 70), (40, 75), (15, 55), (10, 80), (40, 105),
            (90, 105), (140, 85), (180, 75), (200, 80)]
    c.poly(left, brown)
    c.poly([(400 - x, y) for x, y in left], brown)
    c.ellipse((200, 55), (30, 28), brown)
    c.save("mustache.png")

def crown():
    c = Canvas(500, 300)
    gold = (40, 190, 245)
    dark_gold = (20, 140, 200)
    c.poly([(30, 290), (30, 110), (130, 190), (250, 30), (370, 190), (470, 110), (470, 290)], gold)
    c.poly([(30, 290), (30, 245), (470, 245), (470, 290)], dark_gold)
    for x, y in ((30, 105), (250, 25), (470, 105)):
        c.ellipse((x, y), (22, 22), gold)
    for x, color in ((130, (60, 40, 210)), (250, (190, 90, 30)), (370, (70, 170, 40))):
        c.ellipse((x, 220), (24, 18), color)
    c.save("crown.png")

def dog_ears():
    c = Canvas(800, 360)
    brown = (40, 75, 120)
    pink = (170, 160, 240)
    for cx, angle in ((150, 25), (650, -25)):
        c.ellipse((cx, 190), (95, 165), brown, angle=angle)
        c.ellipse((cx, 205), (55, 115), pink, angle=angle)
    c.save("dog_ears.png")

def dog_nose():
    c = Canvas(220, 160)
    c.ellipse((110, 70), (100, 62), (30, 30, 35))
    c.ellipse((80, 45), (25, 12), (140, 140, 150), alpha=200, angle=-15)
    c.poly([(95, 120), (125, 120), (118, 158), (102, 158)], (30, 30, 35))
    c.save("dog_nose.png")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    sunglasses()
    mustache()
    crown()
    dog_ears()
    dog_nose()
