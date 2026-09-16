import torch
from torch.nn import functional as F

from model import STAR


def build_dataset(samples_per_scene: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    scene_ids = torch.arange(3).repeat_interleave(samples_per_scene)
    features = torch.randn(scene_ids.numel(), 8, generator=generator)
    scene_shift = torch.tensor([-0.8, 0.0, 0.8])
    score = 1.2 * features[:, 0] - 0.8 * features[:, 1] + 0.4 * features[:, 2] + scene_shift[scene_ids]
    score = score + (scene_ids == 2).float() * 0.6 * features[:, 3]
    return features, scene_ids, (score > 0).float()


def main() -> None:
    torch.manual_seed(11)
    features, scene_ids, labels = build_dataset()
    model = STAR(input_dim=features.size(1), num_scenes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(101):
        logits = model(features, scene_ids)
        per_example_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        scene_losses = torch.stack([per_example_loss[scene_ids == scene].mean() for scene in range(3)])
        loss = scene_losses.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f} scene_losses={scene_losses.tolist()}")
    with torch.no_grad():
        probabilities = model(features[:6], scene_ids[:6]).sigmoid()
    print(f"probabilities={probabilities.tolist()}")


if __name__ == "__main__":
    main()