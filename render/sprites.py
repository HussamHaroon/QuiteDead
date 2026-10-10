"""Sprite bank: TheLazyStone post-apocalypse pixel art (local, gitignored)
with automatic fallback to the committed Kenney pack.

TheLazyStone sprites are RPG-style 4-directional animation sheets, so facing
is chosen by direction (down/side/up) with walk-cycle frames while moving —
no arbitrary rotation.
"""
import glob
import os

import pygame

_TLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "thelazystone_postapoc")
_KENNEY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "assets", "sprites", "top-down-shooter", "PNG")

PICKUP_SPRITES = {
    "food": "Canned-food",
    "water": "Canned-soup",
    "meds": "Bandage",
    "gun": "Pistol",
    "ammo": "Ammo-crate_Red",
    "bat": "Bat",
}

# zombie id -> visual type (walker / axe brute / big tank)
ZOMBIE_KINDS = ["Zombie_Small", "Zombie_Axe", "Zombie_Big"]


def _frames_from_sheet(path):
    """Split an animation sheet into a list of frames."""
    import re
    img = pygame.image.load(path).convert_alpha()
    m = re.search(r"Sheet(\d+)", path)
    n = int(m.group(1)) if m else 1
    fw = img.get_width() // n
    return [img.subsurface((i * fw, 0, fw, img.get_height())) for i in range(n)]


class TLSSprites:
    """TheLazyStone 4-direction animated sprites, drawn at native 16px scale."""

    def __init__(self, base, theme=None):
        self.base = base
        self.theme = theme or {}
        self.surv_sheets = {}   # (anim, dir) -> [frames]
        self.zomb_sheets = {}   # (kind, anim, dir) -> [frames]
        self.pickups = {}
        self._load("Idle", "idle")
        self._load("Run", "run")
        for kind in ZOMBIE_KINDS:
            for anim, label in (("idle", "Idle"), ("run", "Walk")):
                for d, dirlabel in (("down", "Down"), ("up", "Up"),
                                    ("side", "Side"), ("sideleft", "Side-left")):
                    paths = sorted(glob.glob(os.path.join(
                        base, "Enemies", kind, f"*_{dirlabel}_{label}-Sheet*.png")))
                    if paths:
                        frames = _frames_from_sheet(paths[0])
                        self.zomb_sheets[(kind, anim, d)] = [
                            pygame.transform.scale_by(f, 2) for f in frames]
        for key, name in PICKUP_SPRITES.items():
            p = os.path.join(base, "Objects", "Pickable", f"{name}.png")
            if os.path.exists(p):
                # 1.5x so items actually read on a 48-tile map
                self.pickups[key] = pygame.transform.scale_by(
                    pygame.image.load(p).convert_alpha(), 1.5)
        self._load_death_sheets()
        self._load_map_tiles()

    def _load_death_sheets(self):
        """Death animations: nothing despawns — agents visibly collapse, then
        the persistent stain takes over."""
        self.zomb_death = {}   # (kind, dir) -> [frames]
        for kind in ZOMBIE_KINDS:
            for d, dirlabel in (("down", "Down"), ("up", "Up"),
                                ("side", "Side"), ("sideleft", "Side-left")):
                paths = sorted(glob.glob(os.path.join(
                    self.base, "Enemies", kind, f"*_{dirlabel}_First-Death-Sheet*.png")))
                if paths:
                    frames = _frames_from_sheet(paths[0])
                    self.zomb_death[(kind, d)] = [
                        pygame.transform.scale_by(f, 2) for f in frames]
        self.surv_death = {}   # dir -> [frames]
        base = os.path.join(self.base, "Character", "Main", "Death")
        for d in ("down", "up", "side", "sideleft"):
            label = "side-left" if d == "sideleft" else d
            paths = sorted(glob.glob(os.path.join(base, f"Character_{label}_death1-Sheet*.png")))
            if paths and "_NoHands" not in paths[0]:
                frames = _frames_from_sheet(paths[0])
                self.surv_death[d] = [pygame.transform.scale_by(f, 2) for f in frames]

    def _load_map_tiles(self):
        """Crop ground/wall tiles from the level's themed tilesets, scaled 2x.

        Each tile id maps to a list of variants; the viewer picks one
        deterministically per tile coordinate so the ground looks organic.
        theme['ground'] selects the ground palette, theme['wall_tint'] tints
        walls/floors for per-level atmosphere.
        """
        ground = self.theme.get("ground", "Green")
        ts_path = os.path.join(self.base, "Tiles", f"Background_{ground}_TileSet.png")
        wall_path = os.path.join(self.base, "Tiles", "Brick-Wall_TileSet.png")
        if not os.path.exists(ts_path):
            ts_path = os.path.join(self.base, "Tiles", "Background_Green_TileSet.png")
        if not (os.path.exists(ts_path) and os.path.exists(wall_path)):
            return
        ts = pygame.image.load(ts_path).convert_alpha()
        brick = pygame.image.load(wall_path).convert_alpha()

        def tile(img, tx, ty):
            return img.subsurface((tx * 16, ty * 16, 16, 16)).copy()

        tint = self.theme.get("wall_tint")
        if tint:
            for group in (
                [tile(brick, 0, 0), tile(brick, 4, 0), tile(brick, 5, 1)],
                [tile(ts, 3, 11), tile(ts, 4, 11), tile(ts, 5, 11)],
                [tile(ts, 4, 11)],
            ):
                for s in group:
                    s.fill(tuple(tint), special_flags=pygame.BLEND_RGB_MULT)

        self.map_tiles = {
            0: [tile(ts, 0, 0), tile(ts, 0, 0), tile(ts, 0, 0), tile(ts, 3, 1)],   # grass + flower decor
            1: [tile(ts, 4, 7), tile(ts, 5, 9), tile(ts, 3, 7), tile(ts, 3, 8),
                tile(ts, 3, 9), tile(ts, 4, 7), tile(ts, 5, 9)],                    # asphalt + markings
            2: [tile(ts, 3, 11), tile(ts, 4, 11), tile(ts, 5, 11)],                 # wooden floors
            3: [tile(brick, 0, 0), tile(brick, 4, 0), tile(brick, 5, 1)],           # brick walls
            4: [tile(ts, 4, 11)],                                                   # doorway (wood)
        }

    def _load(self, folder, anim):
        base = os.path.join(self.base, "Character", "Main", folder)
        for d in ("down", "up", "side", "sideleft"):
            label = "side-left" if d == "sideleft" else d
            paths = sorted(glob.glob(os.path.join(base, f"Character_{label}_{anim}-Sheet*.png")))
            if paths and "_no-hands" not in paths[0]:
                frames = _frames_from_sheet(paths[0])
                self.surv_sheets[(anim, d)] = [
                    pygame.transform.scale_by(f, 2) for f in frames]

    @staticmethod
    def _dir_for(heading):
        import math
        dx, dy = math.cos(heading), math.sin(heading)
        if abs(dx) > abs(dy):
            return ("sideleft",) if dx < 0 else ("side",)
        return ("up",) if dy < 0 else ("down",)

    def survivor(self, heading, moving, frame, skin=0, cache=None):
        anim = "run" if moving else "idle"
        (d,) = self._dir_for(heading)
        frames = self.surv_sheets.get((anim, d)) or self.surv_sheets.get(("idle", d))
        if frames is None:
            return None
        return frames[frame % len(frames)]

    def zombie(self, ztype, heading, moving, frame, cache=None):
        kind = ZOMBIE_KINDS[ztype % len(ZOMBIE_KINDS)]
        anim = "run" if moving else "idle"
        (d,) = self._dir_for(heading)
        frames = (self.zomb_sheets.get((kind, anim, d))
                  or self.zomb_sheets.get((kind, "idle", d))
                  or self.zomb_sheets.get((kind, "idle", "down")))
        if frames is None:
            return None
        return frames[frame % len(frames)]

    def pickup(self, kind):
        return self.pickups.get(kind)

    def prop(self, relpath):
        """Decorative world prop (cars, barrels, benches...) by path under the pack root."""
        if not hasattr(self, "_prop_cache"):
            self._prop_cache = {}
        if relpath not in self._prop_cache:
            p = os.path.join(self.base, relpath.replace("/", os.sep))
            self._prop_cache[relpath] = (pygame.image.load(p).convert_alpha()
                                         if os.path.exists(p) else None)
        return self._prop_cache[relpath]

    @property
    def survivor_frame_count(self):
        return 6


class KenneySprites:
    """Fallback: committed Kenney top-down vector sprites (rotated by heading)."""

    def __init__(self, base):
        skins = ["Man Blue/manBlue_hold.png", "Woman Green/womanGreen_hold.png",
                 "Survivor 1/survivor1_hold.png", "Man Brown/manBrown_hold.png",
                 "Hitman 1/hitman1_hold.png", "Soldier 1/soldier1_hold.png"]
        size = (20, 20)
        self.skins = []
        for rel in skins:
            try:
                img = pygame.image.load(os.path.join(base, rel))
            except (FileNotFoundError, pygame.error):
                img = pygame.image.load(os.path.join(base, skins[0]))
            self.skins.append(pygame.transform.smoothscale(img.convert_alpha(), size))
        self.zombie_img = pygame.transform.smoothscale(
            pygame.image.load(os.path.join(base, "Zombie 1/zoimbie1_hold.png")).convert_alpha(), size)
        self.surv_death = {}
        self.zomb_death = {}

    @staticmethod
    def _rotated(cache, img, heading):
        import math
        dx, dy = math.cos(heading), math.sin(heading)
        deg = -math.degrees(math.atan2(dx, -dy))
        key = int(deg // 5) * 5
        s = cache.get((id(img), key))
        if s is None:
            s = pygame.transform.rotate(img, key)
            cache[(id(img), key)] = s
        return s

    def survivor(self, heading, moving, frame, skin=0, cache=None):
        return self._rotated(cache, self.skins[skin % len(self.skins)], heading)

    @staticmethod
    def _dir_for(heading):
        import math
        dx, dy = math.cos(heading), math.sin(heading)
        if abs(dx) > abs(dy):
            return ("sideleft",) if dx < 0 else ("side",)
        return ("up",) if dy < 0 else ("down",)

    def prop(self, relpath):
        return None

    def zombie(self, ztype, heading, moving, frame, cache=None):
        return self._rotated(cache, self.zombie_img, heading)

    def pickup(self, kind):
        return None


def load_bank(theme=None):
    """Build a sprite bank; theme = {'ground': palette name, 'wall_tint': [r,g,b]}."""
    if os.path.isdir(_TLS_DIR):
        try:
            return TLSSprites(_TLS_DIR, theme or {})
        except Exception:
            pass
    return KenneySprites(_KENNEY_DIR)
