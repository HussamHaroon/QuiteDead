"""Generate procedural game sound effects with numpy — no external files needed.

Creates .wav files in assets/audio/generated/:
  gunshot.wav, zombie_growl.wav, bite.wav, melee_hit.wav,
  pickup.wav, deposit.wav, death.wav, reanimation.wav
"""
import math
import os
import struct
import wave

import numpy as np

SR = 22050  # sample rate
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "assets", "audio", "generated")
os.makedirs(OUT, exist_ok=True)


def _write_wav(filename, samples, volume=0.7):
    """Write float samples [-1,1] as a 16-bit mono wav."""
    samples = np.clip(samples * volume, -1.0, 1.0)
    data = (samples * 32767).astype(np.int16)
    path = os.path.join(OUT, filename)
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(data.tobytes())
    print(f"  {filename} ({len(samples)/SR:.2f}s)")


def _envelope(n, attack=0.01, decay=0.3):
    """Attack-decay envelope."""
    t = np.linspace(0, 1, n)
    a = int(attack * n)
    d = int(decay * n)
    env = np.ones(n)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if d > 0:
        env[-d:] *= np.linspace(1, 0, d)
    return env


def gunshot():
    """Sharp crack + low boom."""
    n = int(SR * 0.25)
    t = np.arange(n) / SR
    # crack: high-freq noise with fast decay
    noise = np.random.default_rng(42).uniform(-1, 1, n)
    crack_decay = np.exp(-t * 60)
    crack = noise * crack_decay * 0.8
    # boom: low sine with medium decay
    boom = np.sin(2 * np.pi * 80 * t) * np.exp(-t * 12) * 0.6
    return crack + boom


def zombie_growl():
    """Low rumbling growl with pitch wobble."""
    n = int(SR * 0.6)
    t = np.arange(n) / SR
    # base growl: low frequency with modulation
    f0 = 55
    wobble = 10 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(f0 + wobble) / SR
    growl = np.sign(np.sin(phase)) * 0.5  # square-ish
    # add noise for texture
    noise = np.random.default_rng(7).uniform(-1, 1, n) * 0.3
    env = _envelope(n, attack=0.15, decay=0.3)
    return (growl + noise) * env * 0.5


def bite():
    """Short snap + wet crunch."""
    n = int(SR * 0.15)
    t = np.arange(n) / SR
    # snap: fast transient
    snap = np.sin(2 * np.pi * 2000 * t) * np.exp(-t * 100) * 0.6
    # crunch: noise burst
    noise = np.random.default_rng(3).uniform(-1, 1, n) * np.exp(-t * 50) * 0.4
    return snap + noise


def melee_hit():
    """Impact thud."""
    n = int(SR * 0.12)
    t = np.arange(n) / SR
    thud = np.sin(2 * np.pi * 150 * t) * np.exp(-t * 35)
    click = np.sin(2 * np.pi * 800 * t) * np.exp(-t * 200) * 0.3
    return thud + click


def pickup():
    """Rising chime."""
    n = int(SR * 0.2)
    t = np.arange(n) / SR
    note1 = np.sin(2 * np.pi * 523 * t) * np.exp(-t * 8) * 0.5  # C5
    note2 = np.sin(2 * np.pi * 784 * (t - 0.05) % SR) * np.exp(-t * 8) * 0.3  # G5
    note2 = np.where(t > 0.05, np.sin(2 * np.pi * 784 * t) * np.exp(-t * 8) * 0.3, 0)
    return note1 + note2


def deposit():
    """Success: two ascending notes."""
    n = int(SR * 0.35)
    t = np.arange(n) / SR
    n1 = np.where(t < 0.15, np.sin(2 * np.pi * 440 * t) * np.exp(-t * 5) * 0.4, 0)
    n2 = np.where(t >= 0.12, np.sin(2 * np.pi * 660 * (t - 0.12)) * np.exp(-(t - 0.12) * 5) * 0.4, 0)
    return n1 + n2


def death():
    """Descending tone."""
    n = int(SR * 0.5)
    t = np.arange(n) / SR
    f = 400 * np.exp(-t * 3)
    phase = 2 * np.pi * np.cumsum(f) / SR
    tone = np.sin(phase) * np.exp(-t * 5) * 0.5
    return tone


def reanimation():
    """Eerie rising drone."""
    n = int(SR * 0.8)
    t = np.arange(n) / SR
    f = 50 + 100 * t / 0.8
    phase = 2 * np.pi * np.cumsum(f) / SR
    drone = np.sin(phase) * _envelope(n, 0.3, 0.3) * 0.4
    noise = np.random.default_rng(11).uniform(-1, 1, n) * _envelope(n, 0.3, 0.3) * 0.1
    return drone + noise


def menu_click():
    """Short tick for menu navigation."""
    n = int(SR * 0.05)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 1200 * t) * np.exp(-t * 150) * 0.5


def menu_select():
    """Confirmation blip."""
    n = int(SR * 0.15)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 880 * t) * np.exp(-t * 25) * 0.5


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(f"Generating sounds -> {OUT}/")
    _write_wav("gunshot.wav", gunshot())
    _write_wav("zombie_growl.wav", zombie_growl())
    _write_wav("bite.wav", bite())
    _write_wav("melee_hit.wav", melee_hit())
    _write_wav("pickup.wav", pickup())
    _write_wav("deposit.wav", deposit())
    _write_wav("death.wav", death())
    _write_wav("reanimation.wav", reanimation())
    _write_wav("menu_click.wav", menu_click())
    _write_wav("menu_select.wav", menu_select())
    print("done")
