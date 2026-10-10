"""Append-only event log. Metrics, replays and plots are all projections of this."""


class EventLog:
    def __init__(self):
        self.events = []

    def add(self, tick, etype, **kw):
        e = {"t": tick, "type": etype}
        e.update(kw)
        self.events.append(e)

    def count(self, etype):
        return sum(1 for e in self.events if e["type"] == etype)

    def tally(self, etype, key):
        out = {}
        for e in self.events:
            if e["type"] == etype:
                out[e[key]] = out.get(e[key], 0) + 1
        return out

    def save(self, path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e) + "\n")
