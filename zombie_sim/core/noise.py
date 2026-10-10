"""Spatial noise field.

Actions deposit a value on their tile (plus half on the 4-neighbours); the field
decays multiplicatively every tick. Listeners read tiles within their hearing
radius with linear distance attenuation, so running and combat pull zombies
from far away while a lone walker is only audible up close.
"""


class NoiseField:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.grid = [0.0] * (w * h)

    def emit(self, x, y, amount):
        tx, ty = int(x), int(y)
        if not (0 <= tx < self.w and 0 <= ty < self.h):
            return
        g = self.grid
        g[ty * self.w + tx] += amount
        for nx, ny in ((tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1)):
            if 0 <= nx < self.w and 0 <= ny < self.h:
                g[ny * self.w + nx] += amount * 0.5

    def decay(self, factor):
        g = self.grid
        for i in range(len(g)):
            v = g[i] * factor
            g[i] = v if v > 0.05 else 0.0

    def strongest(self, x, y, r):
        """Perceived noise (distance-attenuated) and its tile within radius r of (x, y)."""
        tx, ty = int(x), int(y)
        x0, x1 = max(0, tx - r), min(self.w - 1, tx + r)
        y0, y1 = max(0, ty - r), min(self.h - 1, ty + r)
        best_v, best_tile = 0.0, None
        g = self.grid
        w = self.w
        for yy in range(y0, y1 + 1):
            row = yy * w
            for xx in range(x0, x1 + 1):
                v = g[row + xx]
                if v <= 0.0:
                    continue
                d = ((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2) ** 0.5
                if d >= r:
                    continue
                perceived = v * (1.0 - d / r)
                if perceived > best_v:
                    best_v, best_tile = perceived, (xx, yy)
        return best_v, best_tile
