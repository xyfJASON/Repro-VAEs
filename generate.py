import argparse

import torch
from torchvision.utils import save_image

import models


@torch.no_grad()
def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ckpt = torch.load(args.model_path, map_location='cpu')
    model = models.VAE(args.img_channels, args.latent_dim)
    model.load_state_dict(ckpt['model'])
    model.to(device=device)

    if args.mode == 'random':
        sample_z = torch.randn((64, args.latent_dim), device=device)
        X = model.generate(sample_z).cpu()
        X = X.view(-1, args.img_channels, args.img_size, args.img_size)
        save_image(X, args.save_path, nrow=8, normalize=True, value_range=(-1, 1))
    elif args.mode == 'walk':
        sample_z1 = torch.randn((5, args.latent_dim), device=device)
        sample_z2 = torch.randn((5, args.latent_dim), device=device)
        result = []
        for t in torch.linspace(0, 1, 15):
            result.append(model.generate(sample_z1 * t + sample_z2 * (1 - t)).cpu())
        result = torch.stack(result, dim=1).reshape(5 * 15, args.img_channels, args.img_size, args.img_size)
        save_image(result, args.save_path, nrow=15, normalize=True, value_range=(-1, 1))
    elif args.mode == 'walk2d':
        sample_z = torch.randn((4, args.latent_dim), device=device)
        result = []
        for i in torch.linspace(0, 1, 10):
            for j in torch.linspace(0, 1, 10):
                result.append(model.generate((sample_z[0:1] * i + sample_z[1:2] * (1 - i)) * j +
                                             (sample_z[2:3] * i + sample_z[3:4] * (1 - i)) * (1 - j)).cpu())
        result = torch.cat(result, dim=0)
        save_image(result, args.save_path, nrow=10, normalize=True, value_range=(-1, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to the saved model')
    parser.add_argument('--mode', choices=['random', 'walk', 'walk2d'], required=True, help='generation mode. Options: random, walk')
    parser.add_argument('--save_path', type=str, required=True, help='path to save the generated result')
    parser.add_argument('--cpu', action='store_true', help='use cpu instead of cuda')
    # Generator settings
    parser.add_argument('--latent_dim', type=int, required=True, help='dimensionality of latent vector')
    parser.add_argument('--img_size', type=int, default=64, help='size of output images, for cnn-like generators')
    parser.add_argument('--img_channels', type=int, default=3, help='number of channels of output images, for cnn-like generators')
    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
