import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow and p.requires_grad:
                    self.shadow[n] = self.shadow[n] * self.decay + p.data * (1 - self.decay)

    def apply(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n])

    def state_dict(self):
        return {'decay': self.decay, 'shadow': {k: v.clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, sd):
        self.decay = sd['decay']
        self.shadow = {k: v.clone() for k, v in sd['shadow'].items()}
