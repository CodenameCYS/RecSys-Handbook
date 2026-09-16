import torch
from torch import nn


class MMoE(nn.Module):
    def __init__(self, input_dim: int, task_count: int = 2, expert_count: int = 4) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU()) for _ in range(expert_count)]
        )
        self.gates = nn.ModuleList([nn.Linear(input_dim, expert_count) for _ in range(task_count)])
        self.towers = nn.ModuleList([nn.Linear(8, 1) for _ in range(task_count)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
        task_logits = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(features), dim=1)
            task_representation = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
            task_logits.append(tower(task_representation).squeeze(-1))
        return torch.stack(task_logits, dim=1)