import torch
from torch import nn


class ConditionalAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int, num_scenes: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.scene_scale = nn.Embedding(num_scenes, bottleneck_dim)
        self.scene_shift = nn.Embedding(num_scenes, bottleneck_dim)
        nn.init.zeros_(self.scene_scale.weight)
        nn.init.zeros_(self.scene_shift.weight)

    def forward(self, representation: torch.Tensor, scene_ids: torch.Tensor) -> torch.Tensor:
        bottleneck = self.down(representation)
        bottleneck = bottleneck * (1 + self.scene_scale(scene_ids)) + self.scene_shift(scene_ids)
        return representation + self.up(torch.relu(bottleneck))


class SceneAdaptiveMLP(nn.Module):
    def __init__(self, input_dim: int, num_scenes: int) -> None:
        super().__init__()
        self.shared_trunk = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU())
        self.adapter = ConditionalAdapter(hidden_dim=16, bottleneck_dim=4, num_scenes=num_scenes)
        self.head = nn.Linear(16, 1)

    def forward(self, features: torch.Tensor, scene_ids: torch.Tensor) -> torch.Tensor:
        representation = self.shared_trunk(features)
        adapted_representation = self.adapter(representation, scene_ids)
        return self.head(adapted_representation).squeeze(-1)