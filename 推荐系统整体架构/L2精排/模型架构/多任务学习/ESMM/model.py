import torch
from torch import nn


class ESMM(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.ctr_tower = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        self.cvr_tower = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctr_logit = self.ctr_tower(features).squeeze(-1)
        cvr_logit = self.cvr_tower(features).squeeze(-1)
        ctcvr_probability = ctr_logit.sigmoid() * cvr_logit.sigmoid()
        return ctr_logit, cvr_logit, ctcvr_probability