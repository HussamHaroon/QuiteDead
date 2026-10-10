"""Scenario configuration: every tunable number of the simulation lives here."""
from dataclasses import dataclass, asdict, fields

import yaml


@dataclass
class Config:
    # world
    width: int = 64
    height: int = 64
    seed: int = 42
    win_ticks: int = 2400                 # survive this long (one day cycle) to win
    # population
    n_survivors: int = 60
    n_zombies: int = 3
    supply_per_building: float = 0.6      # chance a building contains a crate
    n_supply_outdoor: int = 30
    # perception (survivors)
    vision_range: float = 8.0             # tiles
    vision_fov_deg: float = 110.0
    # survivor body
    survivor_walk: float = 0.10           # tiles per tick
    survivor_run: float = 0.17
    survivor_hp: float = 100.0
    hunger_rate: float = 0.060            # per tick — needs must shape mid-game routes
    thirst_rate: float = 0.080
    hunger_damage: float = 0.06           # hp lost per tick at max hunger/thirst
    # survivor combat
    melee_damage: float = 34.0
    melee_chance: float = 0.55
    melee_cooldown: int = 18
    fight_hp_threshold: float = 60.0      # only stand and fight above this hp
    danger_radius: float = 5.0            # flee when a zombie is closer than this
    # guns (looted, not spawned-with)
    gun_range: float = 6.0
    gun_damage: float = 60.0
    gun_chance: float = 0.75
    gun_cooldown: int = 8
    gun_noise: float = 100.0
    ammo_per_crate: int = 8
    n_guns_spawn: int = 4                 # guns placed on procedural maps (rare)
    n_ammo_spawn: int = 8                 # ammo crates placed on procedural maps
    # zombie body
    zombie_walk: float = 0.055
    zombie_chase_speed: float = 0.115
    zombie_vision_range: float = 6.0
    zombie_vision_fov_deg: float = 130.0
    zombie_detect_chance: float = 0.65    # probabilistic detection per tick
    zombie_memory_ticks: int = 240        # how long a last-known position is kept
    zombie_hp: float = 100.0
    zombie_damage: float = 14.0
    zombie_attack_chance: float = 0.6
    zombie_attack_cooldown: int = 15
    zombie_hearing_radius: float = 9.0
    # infection
    bite_infection_chance: float = 0.65
    incubation_ticks: int = 900           # bite -> turn
    reanimation_delay: int = 150          # death -> reanimate
    reanimation_enabled: bool = True      # the Munz resurrection slider
    # noise field
    noise_decay: float = 0.92             # multiplicative decay per tick
    noise_walk: float = 1.0
    noise_run: float = 12.0
    noise_combat: float = 40.0
    noise_hear_threshold: float = 6.0

    @classmethod
    def load(cls, path=None, overrides=None):
        data = {}
        if path:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)
