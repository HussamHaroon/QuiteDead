"""Uniform-grid spatial hash for fast neighbor queries."""
import math


class SpatialHash:
    def __init__(self, cell_size=4.0):
        self.cell = cell_size
        self.buckets = {}

    def rebuild(self, agents):
        self.buckets.clear()
        c = self.cell
        for a in agents:
            if a.alive:
                self.buckets.setdefault((int(a.x // c), int(a.y // c)), []).append(a)

    def query(self, x, y, r):
        """All agents within r of (x, y). Slight over-approximation at cell edges is fine."""
        c = self.cell
        out = []
        for cx in range(int((x - r) // c), int((x + r) // c) + 1):
            for cy in range(int((y - r) // c), int((y + r) // c) + 1):
                b = self.buckets.get((cx, cy))
                if b:
                    out.extend(b)
        return out
