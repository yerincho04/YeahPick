"""Make matched digital-clock PNG tiles with only the displayed time changing.

Uses only the Python standard library. Example:
    python time_clock_assets/digital_clock_png.py \
        --times "6:00 AM" "9:00 AM" "6:00 PM" "9:00 PM"
"""

from __future__ import annotations

import argparse
import math
import re
import struct
import zlib
from pathlib import Path


SIZE = 320
SCALE = 3  # supersampling for clean digit edges
BACKGROUND = (255, 255, 255)
INK = (36, 39, 43)
SEGMENTS = {
    "0": "abcedf", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgecd", "7": "abc", "8": "abcdefg", "9": "abfgcd",
}
LETTERS = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
}


class Canvas:
    def __init__(self) -> None:
        self.width = SIZE * SCALE
        self.pixels = bytearray(bytes(BACKGROUND) * (self.width * self.width))

    def polygon(self, points: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
        pts = [(int(round(x * SCALE)), int(round(y * SCALE))) for x, y in points]
        y0 = max(0, min(y for _, y in pts))
        y1 = min(self.width - 1, max(y for _, y in pts))
        for y in range(y0, y1 + 1):
            crossings = []
            for (x1, py1), (x2, py2) in zip(pts, pts[1:] + pts[:1]):
                if py1 <= y < py2 or py2 <= y < py1:
                    crossings.append(x1 + (y + 0.5 - py1) * (x2 - x1) / (py2 - py1))
            crossings.sort()
            for left, right in zip(crossings[::2], crossings[1::2]):
                x0 = max(0, math.ceil(left))
                x1 = min(self.width, math.ceil(right))
                if x1 > x0:
                    start = (y * self.width + x0) * 3
                    self.pixels[start:start + (x1 - x0) * 3] = bytes(color) * (x1 - x0)

    def png(self, path: Path) -> None:
        # Average the supersampled RGB pixels and write a standards-compliant PNG.
        rows = bytearray()
        for y in range(SIZE):
            rows.append(0)  # PNG filter: none
            for x in range(SIZE):
                totals = [0, 0, 0]
                for sy in range(SCALE):
                    for sx in range(SCALE):
                        i = (((y * SCALE + sy) * self.width) + x * SCALE + sx) * 3
                        for channel in range(3):
                            totals[channel] += self.pixels[i + channel]
                rows.extend(round(total / (SCALE * SCALE)) for total in totals)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

        path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
            + chunk(b"IEND", b"")
        )


def local_point(x: float, y: float) -> tuple[float, float]:
    """Place the display squarely on the white canvas."""
    return 160 + x, 160 + y


def rect(canvas: Canvas, x: float, y: float, w: float, h: float,
         color: tuple[int, int, int]) -> None:
    canvas.polygon([local_point(px, py) for px, py in
                    ((x, y), (x + w, y), (x + w, y + h), (x, y + h))], color)


def digit(canvas: Canvas, symbol: str, x: int, y: int) -> None:
    # Solid seven-segment strokes without boxes around inactive segments.
    parts = {
        "a": (x + 5, y, 29, 5), "g": (x + 5, y + 29, 29, 5),
        "d": (x + 5, y + 58, 29, 5),
        "f": (x, y + 5, 5, 24), "b": (x + 34, y + 5, 5, 24),
        "e": (x, y + 34, 5, 24), "c": (x + 34, y + 34, 5, 24),
    }
    for name, (px, py, w, h) in parts.items():
        if name in SEGMENTS[symbol]:
            rect(canvas, px, py, w, h, INK)


def letter(canvas: Canvas, symbol: str, x: int, y: int) -> None:
    for row, pattern in enumerate(LETTERS[symbol]):
        for col, bit in enumerate(pattern):
            if bit == "1":
                rect(canvas, x + col * 4, y + row * 4, 4, 4, INK)


def parse_time(value: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"\s*(1[0-2]|[1-9]):([0-5][0-9])\s*([AaPp][Mm])\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(f"Use 12-hour time such as '9:00 AM': {value!r}")
    return int(match[1]), int(match[2]), match[3].upper()


def make_clock(hour: int, minute: int, period: str, path: Path) -> None:
    canvas = Canvas()
    # One black outline, a pink bezel, and an unoutlined light-gray screen.
    rect(canvas, -147, -69, 294, 138, INK)
    rect(canvas, -142, -64, 284, 128, (218, 139, 193))
    rect(canvas, -129, -52, 258, 104, (230, 232, 230))

    # Single-digit hours omit the leading zero, as in the reference image.
    if hour < 10:
        digit(canvas, str(hour), -86, -17)
        colon_x, minute_x = -32, (1, 54)
    else:
        digit(canvas, str(hour // 10), -111, -17)
        digit(canvas, str(hour % 10), -65, -17)
        colon_x, minute_x = -17, (13, 60)
    rect(canvas, colon_x, 2, 6, 6, INK)
    rect(canvas, colon_x, 25, 6, 6, INK)
    for symbol, x in zip(f"{minute:02d}", minute_x):
        digit(canvas, symbol, x, -17)
    letter(canvas, period[0], 69, -46)
    letter(canvas, "M", 93, -46)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.png(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--times", nargs="+", default=["6:00 AM", "9:00 AM", "6:00 PM", "9:00 PM"])
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("clock_output"))
    args = parser.parse_args()
    for value in args.times:
        hour, minute, period = parse_time(value)
        path = args.out_dir / f"clock_{hour:02d}-{minute:02d}_{period}.png"
        make_clock(hour, minute, period, path)
        print(path)


if __name__ == "__main__":
    main()
