"""QuiteDead main menu: Simulation / Settings / Quit.

  python -m render.menu   (or launched by run_sim.py with no args)
"""
import os
import sys

import pygame
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TILE = 16
W, H = 640, 480
FPS = 30

BG = (12, 14, 18)
PANEL = (24, 28, 36)
ACCENT = (205, 80, 60)
ACCENT_HI = (255, 120, 90)
TEXT = (220, 220, 220)
TEXT_DIM = (140, 140, 150)

MENU_ITEMS = ["Simulation", "Settings", "Quit"]


def _load_levels():
    from zombie_sim.core.levels import load_levels
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config", "levels.yaml")
    return load_levels(p)


def run_menu():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("QuiteDead")
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("consolas", 48, bold=True)
    font = pygame.font.SysFont("consolas", 24)
    font_small = pygame.font.SysFont("consolas", 14)

    levels = _load_levels()
    selected = 0
    level_index = 1  # default: Parking Lot
    settings = {"speed": 1, "survivors": 6, "show_cone": True, "show_radar": True}
    in_settings = False
    setting_items = ["Level", "Survivors", "Speed", "Vision Cone", "Item Radar", "Back"]
    setting_sel = 0

    done = False
    action = None
    while not done:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                action = "quit"
                done = True
            elif ev.type == pygame.KEYDOWN:
                if not in_settings:
                    if ev.key == pygame.K_UP:
                        selected = (selected - 1) % len(MENU_ITEMS)
                    elif ev.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(MENU_ITEMS)
                    elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                        if MENU_ITEMS[selected] == "Quit":
                            action = "quit"
                            done = True
                        elif MENU_ITEMS[selected] == "Settings":
                            in_settings = True
                        elif MENU_ITEMS[selected] == "Simulation":
                            action = "sim"
                            done = True
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
                            settings["survivors"] = 2 if settings["survivors"] >= 8 \
                                else settings["survivors"] + 2
                        elif name == "Speed":
                            settings["speed"] = {1: 0.5, 0.5: 1, 2: 3, 3: 8, 8: 2}[
                                settings["speed"]]
                        elif name == "Vision Cone":
                            settings["show_cone"] = not settings["show_cone"]
                        elif name == "Item Radar":
                            settings["show_radar"] = not settings["show_radar"]
                    elif ev.key == pygame.K_LEFT:
                        name = setting_items[setting_sel]
                        if name == "Level":
                            level_index = (level_index - 1) % len(levels)
                        elif name == "Survivors":
                            settings["survivors"] = 8 if settings["survivors"] <= 2 \
                                else settings["survivors"] - 2
                    elif ev.key == pygame.K_ESCAPE:
                        in_settings = False

        screen.fill(BG)

        # title
        title = font_big.render("QuiteDead", True, ACCENT)
        screen.blit(title, title.get_rect(center=(W // 2, 80)))
        sub = font_small.render("zombie apocalypse agent simulator", True, TEXT_DIM)
        screen.blit(sub, sub.get_rect(center=(W // 2, 118)))

        if not in_settings:
            for i, item in enumerate(MENU_ITEMS):
                color = ACCENT_HI if i == selected else TEXT
                t = font.render(item, True, color)
                y = 200 + i * 50
                if i == selected:
                    marker = font.render(">", True, ACCENT_HI)
                    screen.blit(marker, (W // 2 - 100, y))
                screen.blit(t, t.get_rect(center=(W // 2, y)))
        else:
            panel = pygame.Surface((W - 80, H - 200))
            panel.fill(PANEL)
            screen.blit(panel, (40, 160))
            for i, name in enumerate(setting_items):
                y = 180 + i * 40
                if name == "Level":
                    val = levels[level_index].name
                elif name == "Survivors":
                    val = str(settings["survivors"])
                elif name == "Speed":
                    val = f"{settings['speed']}x"
                elif name == "Vision Cone":
                    val = "ON" if settings["show_cone"] else "OFF"
                elif name == "Item Radar":
                    val = "ON" if settings["show_radar"] else "OFF"
                else:
                    val = ""
                color = ACCENT_HI if i == setting_sel else TEXT
                label = font.render(f"{name:<16}", True, color)
                value = font.render(f"{val:>12}", True, TEXT)
                screen.blit(label, (70, y))
                screen.blit(value, (W - 160, y))

        hint = font_small.render("ARROWS navigate | ENTER select | ESC back", True, TEXT_DIM)
        screen.blit(hint, hint.get_rect(center=(W // 2, H - 30)))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    return action, {"level_index": level_index, **settings}


if __name__ == "__main__":
    result, cfg = run_menu()
    print(f"action: {result}, config: {cfg}")
