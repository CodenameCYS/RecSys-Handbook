import torch
from torch import nn


class Expert(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class PLE(nn.Module):
    def __init__(self, input_dim: int, task_count: int = 2, task_expert_count: int = 2, shared_expert_count: int = 2) -> None:
        super().__init__()
        self.task_experts = nn.ModuleList(
            [nn.ModuleList([Expert(input_dim) for _ in range(task_expert_count)]) for _ in range(task_count)]
        )
        self.shared_experts = nn.ModuleList([Expert(input_dim) for _ in range(shared_expert_count)])
        self.gates = nn.ModuleList([nn.Linear(input_dim, task_expert_count + shared_expert_count) for _ in range(task_count)])
        self.towers = nn.ModuleList([nn.Linear(8, 1) for _ in range(task_count)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shared_outputs = [expert(features) for expert in self.shared_experts]
        task_logits = []
        for experts, gate, tower in zip(self.task_experts, self.gates, self.towers):
            expert_outputs = torch.stack([*(expert(features) for expert in experts), *shared_outputs], dim=1)
            weights = torch.softmax(gate(features), dim=1)
            representation = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
            task_logits.append(tower(representation).squeeze(-1))
        return torch.stack(task_logits, dim=1)