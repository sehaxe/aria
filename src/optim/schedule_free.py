import torch

class ScheduleFreeWrapper:
    def __init__(self, base_optimizer, beta=0.9):
        self.base = base_optimizer
        self.beta = beta
        self._state = []

    def _flat(self):
        return [p for g in self.base.param_groups for p in g['params']]

    def _ensure(self, idx, p):
        while len(self._state) <= idx:
            self._state.append(None)
        if self._state[idx] is None:
            self._state[idx] = {'x': p.data.clone(), 'z': p.data.clone(), 't': 0}
        return self._state[idx]

    @torch.no_grad()
    def interpolate_params(self):
        for idx, p in enumerate(self._flat()):
            s = self._ensure(idx, p)
            s['z'] = p.data.clone()
            p.data.lerp_(s['x'], 1 - self.beta)

    @torch.no_grad()
    def step(self):
        self.base.step()
        for idx, p in enumerate(self._flat()):
            s = self._state[idx] if idx < len(self._state) else None
            if s is None or p.grad is None:
                continue
            t = s['t'] + 1
            s['x'].lerp_(p.data, 1.0 / t)
            s['t'] = t
            p.data.lerp_(s['x'], 1 - self.beta)

    @torch.no_grad()
    def eval_params(self):
        for idx, p in enumerate(self._flat()):
            s = self._state[idx] if idx < len(self._state) else None
            if s is not None:
                p.data.copy_(s['x'])
