import torch
from torch.nn import functional as F

from model import ESMM


def build_dataset(size: int = 512) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(size, 8, generator=generator)
    clicks = ((features[:, 0] + features[:, 1]) > 0).float()
    conversions = (clicks * ((features[:, 2] - features[:, 3]) > 0).float())
    return features, clicks, conversions


def main() -> None:
    torch.manual_seed(7)
    features, clicks, conversions = build_dataset()
    model = ESMM(input_dim=features.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(101):
        ctr_logit, _, ctcvr_probability = model(features)
        ctr_loss = F.binary_cross_entropy_with_logits(ctr_logit, clicks)
        ctcvr_loss = F.binary_cross_entropy(ctcvr_probability, conversions)
        loss = ctr_loss + ctcvr_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f} ctr={ctr_loss.item():.4f} ctcvr={ctcvr_loss.item():.4f}")
    with torch.no_grad():
        _, _, probabilities = model(features[:5])
    print(f"ctcvr_probabilities={probabilities.tolist()}")


if __name__ == "__main__":
    main()