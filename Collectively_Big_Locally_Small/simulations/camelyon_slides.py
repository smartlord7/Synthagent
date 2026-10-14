"""Reading CAMELYON whole-slide images from the official release by HTTP range requests (no full-slide download).

The CAMELYON16/17 slides (AWS Open Data, https://camelyon-dataset.s3.amazonaws.com) are tiled pyramidal (Big)TIFFs
written by ASAP; one slide is 1-4 GB and the 50 annotated CAMELYON17 slides about 150 GB, more than this machine's disk.
`Slide` parses the TIFF directories through a block-cached HTTP file object and fetches only the tiles a region needs,
so a 96 x 96 patch at level 2 costs one to four tile requests and a whole level-4 image a few tens of MB.

Levels are the TIFF pages of full-resolution type (subfiletype 0 or reduced-resolution pages of the main series), in
decreasing size, as OpenSlide reports them for these files (level_dimensions[k] = (width, height) of page k).
"""
import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import tifffile

BASE = "https://camelyon-dataset.s3.amazonaws.com"


class HTTPFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a block cache (for the TIFF directories)."""

    def __init__(self, url, block=1 << 18, session=None):
        self.url, self.block, self.pos = url, block, 0
        self.s = session or requests.Session()
        r = self._req("HEAD")
        self.size = int(r.headers["Content-Length"])
        self.cache = {}

    def _req(self, method, rng=None):
        for t in range(6):
            try:
                r = self.s.request(method, self.url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {},
                                   timeout=120)
                r.raise_for_status()
                return r
            except requests.RequestException:
                if t == 5:
                    raise
                time.sleep(2 ** t)

    def range(self, a, n):
        """n bytes from offset a, uncached (tile data)."""
        return self._req("GET", (a, a + n - 1)).content

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _blk(self, b):
        if b not in self.cache:
            if len(self.cache) > 256:
                self.cache.clear()
            a = b * self.block
            self.cache[b] = self.range(a, min(self.block, self.size - a))
        return self.cache[b]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            b, o = divmod(self.pos, self.block)
            chunk = self._blk(b)[o:o + n - len(out)]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[:len(d)] = d
        return len(d)


class Slide:
    def __init__(self, key, session=None):
        self.f = HTTPFile(f"{BASE}/{key}", session=session)
        self.tif = tifffile.TiffFile(self.f)
        pages = [p for p in self.tif.pages if p.is_tiled and p.samplesperpixel >= 3]
        pages.sort(key=lambda p: -p.imagewidth)
        self.pages = pages
        self.level_dimensions = [(p.imagewidth, p.imagelength) for p in pages]

    def _tile(self, page, ty, tx):
        tw, tl = page.tilewidth, page.tilelength
        ntx = -(-page.imagewidth // tw)
        idx = ty * ntx + tx
        off, cnt = page.dataoffsets[idx], page.databytecounts[idx]
        if cnt == 0:
            return np.full((tl, tw, 3), 255, np.uint8)
        data = self.f.range(off, cnt)
        a = page.decode(data, idx, jpegtables=page.jpegtables)[0]
        return np.asarray(a).reshape(tl, tw, -1)[:, :, :3]

    def read_level(self, level, x, y, w, h):
        """RGB region (h, w, 3) of `level` with top-left (x, y) in that level's pixels; outside the slide is white
        (OpenSlide returns transparent there, which WILDS converts to black via RGBA->RGB; such regions do not occur
        for the patches used, which lie at least 5 grid cells inside the slide)."""
        p = self.pages[level]
        tw, tl = p.tilewidth, p.tilelength
        out = np.full((h, w, 3), 255, np.uint8)
        tiles = [(ty, tx) for ty in range(max(0, y // tl), min(-(-p.imagelength // tl), (y + h - 1) // tl + 1))
                 for tx in range(max(0, x // tw), min(-(-p.imagewidth // tw), (x + w - 1) // tw + 1))]
        with ThreadPoolExecutor(min(16, len(tiles) or 1)) as ex:
            got = list(ex.map(lambda t: self._tile(p, *t), tiles))
        for (ty, tx), t in zip(tiles, got):
            x0, y0 = tx * tw, ty * tl
            ax, ay = max(x, x0), max(y, y0)
            bx, by = min(x + w, x0 + tw, p.imagewidth), min(y + h, y0 + tl, p.imagelength)
            if bx > ax and by > ay:
                out[ay - y:by - y, ax - x:bx - x] = t[ay - y0:by - y0, ax - x0:bx - x0]
        return out

    def whole_level(self, level):
        w, h = self.level_dimensions[level]
        return self.read_level(level, 0, 0, w, h)
