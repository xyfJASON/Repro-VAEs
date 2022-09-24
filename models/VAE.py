import torch
import torch.nn as nn
from torch import Tensor


class VAE(nn.Module):
    def __init__(self, img_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(img_channels, latent_dim)
        self.decoder = Decoder(img_channels, latent_dim)

    def forward(self, X: Tensor):
        """
        Args:
            X (Tensor): [bs, C, H, W]
        Returns:
            out (Tensor): [bs, C, H, W]
            mean (Tensor): [bs, D]
            logvar (Tensor): [bs, D]
        """
        mean, logvar = self.encoder(X)
        z = torch.randn_like(mean) * torch.exp(logvar / 2) + mean
        out = self.decoder(z)
        return out, mean, logvar

    def generate(self, z: Tensor):
        return self.decoder(z)

    def reconstruct(self, X: Tensor):
        return self.forward(X)[0]


class Encoder(nn.Module):
    def __init__(self, img_channels: int, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, (4, 4), stride=(2, 2), padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=1),           # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=1),          # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=1),          # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, (4, 4), stride=(1, 1), padding=0),          # 1x1
            nn.Flatten(),
        )
        self.fc_mean = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, X: Tensor):
        X = self.encoder(X)
        mean = self.fc_mean(X)
        logvar = self.fc_logvar(X)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, img_channels: int, latent_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, (4, 4), stride=(1, 1), padding=(0, 0)),            # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, (4, 4), stride=(2, 2), padding=(1, 1)),            # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, (4, 4), stride=(2, 2), padding=(1, 1)),            # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, (4, 4), stride=(2, 2), padding=(1, 1)),             # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, img_channels, (4, 4), stride=(2, 2), padding=(1, 1)),    # 64x64
            nn.Tanh(),
        )

    def forward(self, z: Tensor):
        z = self.proj(z)
        z = z.view(-1, z.shape[1], 1, 1)
        out = self.decoder(z)
        return out


def _test():
    vae = VAE()
    X = torch.rand((10, 3, 64, 64))
    out, mean, logvar = vae(X)
    print(out.shape, mean.shape, logvar.shape)


if __name__ == '__main__':
    _test()
