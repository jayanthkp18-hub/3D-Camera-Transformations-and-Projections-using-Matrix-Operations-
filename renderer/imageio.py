"""Tiny dependency-free PNG/PPM writers (so the demo needs only numpy)."""

import struct
import zlib
import numpy as np


def save_png(path: str, image: np.ndarray) -> None:
    img = np.ascontiguousarray(image, dtype=np.uint8)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def save_ppm(path: str, image: np.ndarray) -> None:
    h, w, _ = image.shape
    with open(path, "wb") as f:
        f.write(f"P6 {w} {h} 255\n".encode())
        f.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
