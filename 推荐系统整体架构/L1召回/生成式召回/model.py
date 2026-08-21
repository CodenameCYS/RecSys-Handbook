import torch
from torch import nn
from torch.nn import functional as F


class GenerativeRecallModel(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int = 32, hidden_dim: int = 48) -> None:
        super().__init__()
        self.item_embeddings = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_items)

    def forward(self, item_sequences: torch.Tensor) -> torch.Tensor:
        embeddings = self.item_embeddings(item_sequences)
        hidden_states, _ = self.encoder(embeddings)
        return self.output(hidden_states)

    def next_item_loss(self, item_sequences: torch.Tensor) -> torch.Tensor:
        inputs = item_sequences[:, :-1]
        targets = item_sequences[:, 1:]
        logits = self(inputs)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    @torch.no_grad()
    def next_items(self, history: torch.Tensor, top_k: int = 5) -> list[int]:
        logits = self(history)[:, -1, :]
        logits[:, 0] = float("-inf")
        return logits.topk(top_k, dim=-1).indices.squeeze(0).tolist()