"""Regenerate config/default.yaml from the Config dataclass (single source of truth)."""
import os, sys
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from zombie_sim.core.config import Config  # noqa: E402

hdr = ("# QuiteDead default scenario. GENERATED from the Config dataclass defaults\n"
       "# (python scripts/gen_default_yaml.py) so YAML and code can never drift apart.\n")
with open(os.path.join(ROOT, "config", "default.yaml"), "w", encoding="utf-8") as f:
    f.write(hdr + yaml.safe_dump(Config().to_dict(), sort_keys=False))
