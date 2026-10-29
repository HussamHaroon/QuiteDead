"""Squad communication + utility_ai brain: determinism and basic sanity."""
import pytest
from zombie_sim.core.config import Config
from zombie_sim.core.sim import World


def _run(brain, seed=3, ticks=300):
    w = World(Config(**{**Config().to_dict(), "seed": seed, "brain_survivors": brain}))
    for _ in range(ticks):
        w.step()
    return [(round(s.x, 3), round(s.y, 3)) for s in w.survivors]


@pytest.mark.parametrize("brain", ["squad", "utility_ai", "tactical"])
def test_deterministic(brain):
    assert _run(brain) == _run(brain)
