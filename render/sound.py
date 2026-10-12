"""Sound manager: event-driven sound effects for the simulation.

Loads procedurally generated .wav files and plays them when game events
occur (shots, bites, deaths, pickups, etc.). Called from the viewer's
event scanner — the core sim stays audio-free.
"""
import os
import threading

import pygame

SOUND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "audio", "generated")

# event type -> sound file
EVENT_SOUNDS = {
    "shot": "gunshot.wav",
    "bite": "bite.wav",
    "kill": "melee_hit.wav",           # zombie killed by survivor
    "killz": "death.wav",              # survivor killed by zombie
    "death": "death.wav",
    "turn": "reanimation.wav",         # survivor reanimates
    "loot": "pickup.wav",
    "deposit": "deposit.wav",
    "reinforce": "zombie_growl.wav",
}


class SoundManager:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.sounds = {}
        self._channel_idx = 0
        self._channels = [pygame.mixer.Channel(i) for i in range(8)]
        self.load()

    def load(self):
        if not self.enabled:
            return
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        for event, filename in EVENT_SOUNDS.items():
            path = os.path.join(SOUND_DIR, filename)
            if os.path.exists(path):
                try:
                    self.sounds[event] = pygame.mixer.Sound(path)
                except pygame.error:
                    pass

    def play(self, event_type):
        if not self.enabled or event_type not in self.sounds:
            return
        snd = self.sounds[event_type]
        ch = self._channels[self._channel_idx % len(self._channels)]
        self._channel_idx += 1
        ch.play(snd)

    def play_menu(self, name):
        """Menu sounds: click / select."""
        path = os.path.join(SOUND_DIR, f"menu_{name}.wav")
        if os.path.exists(path):
            try:
                pygame.mixer.Sound(path).play()
            except pygame.error:
                pass

    def stop_all(self):
        pygame.mixer.stop()
