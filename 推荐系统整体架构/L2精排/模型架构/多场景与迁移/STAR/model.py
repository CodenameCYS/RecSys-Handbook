import torch
from torch import nn


class StarLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_scenes: int) -> None:
        super().__init__()
        self.shared_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.shared_bias = nn.Parameter(torch.zeros(output_dim))
        self.scene_weight = nn.Embedding(num_scenes, output_dim * input_dim)
        self.scene_bias = nn.Embedding(num_scenes, output_dim)
        nn.init.xavier_uniform_(self.shared_weight)
        nn.init.ones_(self.scene_weight.weight)
        nn.init.zeros_(self.scene_bias.weight)

    def forward(self, features: torch.Tensor, scene_ids: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        scene_weight = self.scene_weight(scene_ids).view(batch_size, *self.shared_weight.shape)
        effective_weight = self.shared_weight.unsqueeze(0) * scene_weight
        output = torch.bmm(effective_weight, features.unsqueeze(-1)).squeeze(-1)
        return output + self.shared_bias + self.scene_bias(scene_ids)


class STAR(nn.Module):
    def __init__(self, input_dim: int, num_scenes: int) -> None:
        super().__init__()
        self.hidden = StarLinear(input_dim, 16, num_scenes)
        self.output = StarLinear(16, 1, num_scenes)

    def forward(self, features: torch.Tensor, scene_ids: torch.Tensor) -> torch.Tensor:
        representation = torch.relu(self.hidden(features, scene_ids))
        return self.output(representation, scene_ids).squeeze(-1)