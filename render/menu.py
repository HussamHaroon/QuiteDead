"""QuiteDead main menu with TLS assets: buttons, background, decoration."""
import math
import os
import sys

import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W, H = 800, 600
FPS = 30
ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "assets", "thelazystone_postapoc")


def _load(path):
    p = os.path.join(ASSETS, path)
    return pygame.image.load(p).convert_alpha() if os.path.exists(p) else None


def _load_menu_bg(w, h):
    """Load the custom menu background if available, else generate tiles."""
    custom = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "menu_assets",
                          "Gemini_Generated_Image_oflwsroflwsroflw.jpg")
    if os.path.exists(custom):
        img = pygame.image.load(custom)
        iw, ih = img.get_size()
        scale = max(w / iw, h / ih)
        img = pygame.transform.scale(img, (int(iw * scale), int(ih * scale)))
        ox = (img.get_width() - w) // 2
        oy = (img.get_height() - h) // 2
        return img.subsurface((ox, oy, w, h))
    return _tile_background(w, h)


def _tile_background(w, h):
    """Create a post-apocalyptic tile background from the TLS ground tiles."""
    ts = _load("Tiles/Background_Dark-Green_TileSet.png")
    brick = _load("Tiles/Brick-Wall_TileSet.png")
    if ts is None:
        return pygame.Surface((w, h))
    surf = pygame.Surface((w, h))
    # fill with dark grass (tile 0,0) repeated
    grass = ts.subsurface((0, 0, 16, 16))
    for y in range(0, h, 16):
        for x in range(0, w, 16):
            surf.blit(grass, (x, y))
    # darken
    dark = pygame.Surface((w, h), pygame.SRCALPHA)
    dark.fill((0, 0, 0, 140))
    surf.blit(dark, (0, 0))
    # scatter some road/brick patches for texture
    import random
    rng = random.Random(42)
    road = ts.subsurface((4 * 16, 7 * 16, 16, 16))
    wall = brick.subsurface((0, 0, 16, 16)) if brick else road
    for _ in range(30):
        tx, ty = rng.randrange(0, w - 32, 16), rng.randrange(0, h - 32, 16)
        tile = road if rng.random() < 0.5 else wall
        surf.blit(tile, (tx, ty))
    surf.blit(dark, (0, 0))  # darken again after tiles
    return surf


def _button(text, pressed=False, font=None):
    """Render a TLS menu button with text overlay."""
    state = "Pressed" if pressed else "Not-Pressed"
    img = _load(f"UI/Menu/Main Menu/Blank_{state}.png")
    if img is None:
        img = _load(f"UI/Menu/Main Menu/Play_{state}.png")
    if img is None:
        # fallback: simple rect
        surf = pygame.Surface((200, 50))
        surf.fill((180, 60, 50) if pressed else (120, 40, 35))
        if font:
            t = font.render(text, True, (255, 255, 255))
            surf.blit(t, t.get_rect(center=surf.get_rect().center))
        return surf
    # scale to a reasonable button size
    img = pygame.transform.scale(img, (220, 52))
    if font:
        t = font.render(text, True, (230, 230, 230))
        img.blit(t, t.get_rect(center=img.get_rect().center))
    return img


def run_menu():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("QuiteDead")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("consolas", 52, bold=True)
    font = pygame.font.SysFont("consolas", 22)
    font_sm = pygame.font.SysFont("consolas", 14)

    bg = _load_menu_bg(W, H)

    # decorative agents: a zombie chasing a survivor across the bottom
    if zombie_img:
        zombie_img = pygame.transform.scale_by(
            zombie_img.subsurface((0, 0, zombie_img.get_width() // 6,
                                   zombie_img.get_height())), 4)
    if survivor_img:
        survivor_img = pygame.transform.scale_by(
            survivor_img.subsurface((0, 0, survivor_img.get_width() // 6,
                                     survivor_img.get_height())), 4)

    levels = _load_levels()
    selected = 0
    level_index = 1
    n_survivors = 6
    speed = 1
    show_cone = True
    show_radar = True
    in_settings = False
    setting_sel = 0
    setting_items = ["Level", "Survivors", "Speed", "Vision Cone", "Item Radar", "Back"]

    menu_items = ["Simulation", "Settings", "Quit"]
    done = False
    action = None
    t = 0.0

    while not done:
        t += 0.033
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                action = "quit"
                done = True
            elif ev.type == pygame.KEYDOWN:
                if not in_settings:
                    if ev.key == pygame.K_UP:
                        selected = (selected - 1) % len(menu_items)
                    elif ev.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(menu_items)
                    elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                        if menu_items[selected] == "Quit":
                            action = "quit"; done = True
                        elif menu_items[selected] == "Settings":
                            in_settings = True
                        else:
                            action = "sim"; done = True
                else:
                    if ev.key == pygame.K_UP:
                        setting_sel = (setting_sel - 1) % len(setting_items)
                    elif ev.key == pygame.K_DOWN:
                        setting_sel = (setting_sel + 1) % len(setting_items)
                    elif ev.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_RIGHT):
                        name = setting_items[setting_sel]
                        if name == "Back":
                            in_settings = False
                        elif name == "Level":
                            level_index = (level_index + 1) % len(levels)
                        elif name == "Survivors":
                            n_survivors = 2 if n_survivors >= 8 else n_survivors + 2
                        elif name == "Speed":
                            speed = {1: 0.5, 0.5: 1, 2: 3, 3: 8, 8: 2}[speed]
                        elif name == "Vision Cone":
                            show_cone = not show_cone
                        elif name == "Item Radar":
                            show_radar = not show_radar
                    elif ev.key == pygame.K_LEFT:
                        name = setting_items[setting_sel]
                        if name == "Level":
                            level_index = (level_index - 1) % len(levels)
                        elif name == "Survivors":
                            n_survivors = 8 if n_survivors <= 2 else n_survivors - 2
                    elif ev.key == pygame.K_ESCAPE:
                        in_settings = False

        screen.blit(bg, (0, 0))

        # title with glow
        glow = pygame.font.SysFont("consolas", 56, bold=True)
        title_glow = glow.render("QuiteDead", True, (120, 30, 20))
        title = font_big.render("QuiteDead", True, (220, 70, 50))
        screen.blit(title_glow, title_glow.get_rect(center=(W // 2 + 2, 82)))
        screen.blit(title, title.get_rect(center=(W // 2, 80)))
        sub = font_sm.render("zombie apocalypse agent simulator", True, (180, 170, 160))
        screen.blit(sub, sub.get_rect(center=(W // 2, 116)))

        if not in_settings:
            for i, item in enumerate(menu_items):
                pressed = (i == selected)
                btn = _button(item, pressed=pressed, font=font)
                y = 180 + i * 70
                screen.blit(btn, btn.get_rect(center=(W // 2, y)))
        else:
            # settings panel
            panel = pygame.Surface((W - 120, 320), pygame.SRCALPHA)
            panel.fill((20, 24, 32, 220))
            pygame.draw.rect(panel, (80, 60, 50), panel.get_rect(), 2)
            screen.blit(panel, (60, 160))

            header = font.render("SETTINGS", True, (220, 100, 80))
            screen.blit(header, header.get_rect(center=(W // 2, 185)))

            for i, name in enumerate(setting_items):
                if name == "Level":
                    val = levels[level_index].name
                elif name == "Survivors":
                    val = str(n_survivors)
                elif name == "Speed":
                    val = f"{speed}x"
                elif name == "Vision Cone":
                    val = "ON" if show_cone else "OFF"
                elif name == "Item Radar":
                    val = "ON" if show_radar else "OFF"
                else:
                    val = ""
                y = 215 + i * 36
                sel = (i == setting_sel)
                label_col = (255, 140, 100) if sel else (200, 200, 200)
                label = font.render(f"  {name}", True, label_col)
                value = font.render(f"{val}  ", True, (200, 200, 200) if not sel else (255, 200, 160))
                screen.blit(label, (100, y))
                screen.blit(value, (W - 200, y))

        hint = font_sm.render("ARROWS navigate  |  ENTER select  |  ESC back", True, (150, 140, 130))
        screen.blit(hint, hint.get_rect(center=(W // 2, H - 24)))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    return action, {"level_index": level_index, "survivors": n_survivors,
                    "speed": speed, "show_cone": show_cone, "show_radar": show_radar}


def _load_levels():
    from zombie_sim.core.levels import load_levels
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config", "levels.yaml")
    return load_levels(p)


if __name__ == "__main__":
    a, c = run_menu()
    print(a, c)
