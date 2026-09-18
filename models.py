import torch
from torch import Tensor, nn


class CVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim + num_classes, out_features=hidden_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        self.logvar = nn.Linear(in_features=hidden_dim, out_features=latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim + num_classes, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=input_dim),
        )

    def encode(self, x: Tensor, y: Tensor):
        h = self.encoder(torch.cat([x, y], dim=1))
        return self.mu(h), self.logvar(h)

    def decode(self, z: Tensor, y: Tensor):
        return self.decoder(torch.cat([z, y], dim=1))

    def forward(self, x: Tensor, y: Tensor):
        mu, logvar = self.encode(x, y)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        x_hat = self.decode(z, y)

        return x_hat, mu, logvar
