"""Torch-free loader/inference for rl/ppo.ActorCritic checkpoints (sandbox-only helper)."""
import pickle, zipfile, collections
import numpy as np

class _Storage:
    def __init__(self, arr): self.arr = arr

def load_state_dict(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    def rebuild(storage, offset, size, stride, *a, **k):
        n = int(np.prod(size)) if len(size) else 1
        flat = storage.arr[offset:offset + (max(s*st for s, st in zip(size, stride)) if len(size) else 1)]
        return np.lib.stride_tricks.as_strided(storage.arr[offset:], shape=tuple(size),
                strides=tuple(s * 4 for s in stride)).copy()
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2": return rebuild
            if name == "OrderedDict": return collections.OrderedDict
            if mod == "torch" and name.endswith("Storage"): return name
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            _, st, key, loc, numel = pid
            raw = z.read(f"{root}/data/{key}")
            return _Storage(np.frombuffer(raw, dtype=np.float32))
    return U(z.open(f"{root}/data.pkl")).load()

class NPActorCritic:
    def __init__(self, path, seed=0):
        sd = load_state_dict(path)
        self.w = [(sd[f"trunk.{i}.weight"], sd[f"trunk.{i}.bias"]) for i in (0, 2)]
        self.pi = (sd["pi.weight"], sd["pi.bias"])
        self.rng = np.random.default_rng(seed)
    def act(self, obs, deterministic=False):
        h = np.asarray(obs, dtype=np.float32)
        for W, b in self.w:
            h = np.maximum(h @ W.T + b, 0)
        logits = h @ self.pi[0].T + self.pi[1]
        if deterministic:
            a = logits.argmax(-1)
        else:
            p = np.exp(logits - logits.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
            a = np.array([self.rng.choice(len(r), p=r) for r in p])
        return a, None, None
