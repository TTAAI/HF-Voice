"""Architectures used by the eight fixed R72 experts."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dimension, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(dimension, dimension), nn.Dropout(dropout))

    def forward(self, values):
        return values+self.layers(values)


class NeuralExpert(nn.Module):
    def __init__(self, cfg, speech_tasks, input_dimension=1024):
        super().__init__()
        self.speech_tasks = list(speech_tasks)
        self.separate = cfg['architecture'] == 'residual_separate'
        self.adapter = cfg['architecture'] in ['residual_adapter', 'mlp_adapter']
        self.weighted = cfg['fusion'] == 'task_weight'
        p, h, d = cfg['projection'], cfg['hidden'], cfg['dropout']

        def encoder():
            return nn.Sequential(nn.Linear(input_dimension, p), nn.LayerNorm(p), nn.GELU(), nn.Dropout(d))

        if self.separate:
            self.segment_encoders = nn.ModuleDict({s: encoder() for s in self.speech_tasks})
            self.segment_encoder = None
        else:
            self.segment_encoder = encoder()
            self.segment_encoders = nn.ModuleDict()
        if self.adapter:
            self.modality_adapters = nn.ModuleDict({s: ResidualBlock(p, d) if cfg['architecture'] == 'residual_adapter'
                else nn.Sequential(nn.LayerNorm(p), nn.Linear(p, p), nn.GELU(), nn.Dropout(d)) for s in self.speech_tasks})
        else:
            self.modality_adapters = nn.ModuleDict()
        self.task_weight_logits = nn.Parameter(torch.zeros(len(self.speech_tasks))) if self.weighted else None
        self.fusion = nn.Sequential(nn.Linear(p, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(d))
        self.body = nn.Sequential(ResidualBlock(h, d), ResidualBlock(h, d)) if self.separate else nn.Identity()
        self.head = nn.Linear(h, cfg['classes'])

    def forward(self, inputs, masks):
        pooled = []
        for speech in self.speech_tasks:
            encoder = self.segment_encoders[speech] if self.separate else self.segment_encoder
            values = encoder(inputs[speech])
            if self.adapter:
                values = self.modality_adapters[speech](values)
            mask = masks[speech]
            pooled.append((values*mask.unsqueeze(-1)).sum(dim=1)/mask.sum(dim=1, keepdim=True).clamp_min(1.))
        tasks = torch.stack(pooled, dim=1)
        mean = tasks.mean(dim=1)
        fused = (tasks*torch.softmax(self.task_weight_logits, dim=0).view(1, -1, 1)).sum(dim=1) if self.weighted else mean
        return self.head(self.body(self.fusion(fused)))


class OrdinalExpert(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.score = nn.Linear(dimension, 1)
        self.first = nn.Parameter(torch.tensor(-1.))
        self.raw_steps = nn.Parameter(torch.tensor([.54132485, .54132485]))

    def probabilities(self, values):
        score = self.score(values).squeeze(1)
        thresholds = torch.cat([self.first[None], self.first[None]+torch.cumsum(F.softplus(self.raw_steps), dim=0)])
        cdf = torch.sigmoid(thresholds[None, :]-score[:, None])
        return torch.stack([cdf[:, 0], cdf[:, 1]-cdf[:, 0], cdf[:, 2]-cdf[:, 1], 1.-cdf[:, 2]], dim=1).clamp_min(1e-7)


def expert_model(cfg, speech_tasks):
    if cfg['architecture'] == 'ordinal':
        return OrdinalExpert(cfg['pca_dimension'])
    return NeuralExpert(cfg, speech_tasks)
