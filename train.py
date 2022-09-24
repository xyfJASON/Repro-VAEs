import os
import yaml
import imageio
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

import models
from utils.data import build_dataset, build_dataloader
from utils.optimizer import build_optimizer
from utils.train_utils import reduce_tensor, set_device, create_log_directory


class Trainer:
    def __init__(self, config_path: str):
        # ====================================================== #
        # READ CONFIGURATION FILE
        # ====================================================== #
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # ====================================================== #
        # SET DEVICE
        # ====================================================== #
        self.device, self.world_size, self.local_rank, self.global_rank = set_device()
        self.is_master = self.world_size <= 1 or self.global_rank == 0
        self.is_ddp = self.world_size > 1
        print('using device:', self.device)

        # ====================================================== #
        # CREATE LOG DIRECTORY
        # ====================================================== #
        if self.is_master:
            self.log_root = create_log_directory(self.config, config_path)

        # ====================================================== #
        # TENSORBOARD
        # ====================================================== #
        if self.is_master:
            os.makedirs(os.path.join(self.log_root, 'tensorboard'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))

        # ====================================================== #
        # DATA
        # ====================================================== #
        train_dataset, self.img_channels = build_dataset(self.config['dataset'], dataroot=self.config['dataroot'], img_size=64, split='train')
        self.train_loader = build_dataloader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, is_ddp=self.is_ddp)

        # ====================================================== #
        # BUILD MODELS, OPTIMIZERS AND SCHEDULERS
        # ====================================================== #
        self.model = models.VAE(self.img_channels, self.config['latent_dim'])
        self.model.to(device=self.device)
        self.optimizer = build_optimizer(self.model.parameters(), cfg=self.config['optimizer'])
        # distributed
        if self.is_ddp:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # ====================================================== #
        # TEST SAMPLES
        # ====================================================== #
        self.sample_z = torch.randn((64, self.config['latent_dim']), device=self.device)

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.model.to(device=self.device)

    def save_model(self, model_path):
        model = self.model.module if self.is_ddp else self.model
        torch.save({'model': model.state_dict()}, model_path)

    def train(self):
        print('==> Training...')
        sample_paths = []
        for ep in range(self.config['epochs']):
            if self.is_ddp:
                dist.barrier()
                self.train_loader.sampler.set_epoch(ep)

            self.train_one_epoch(ep)

            if self.is_master:
                if self.config.get('save_freq') and (ep + 1) % self.config['save_freq'] == 0:
                    self.save_model(os.path.join(self.log_root, 'ckpt', f'epoch_{ep}.pt'))
                if self.config.get('sample_freq') and (ep + 1) % self.config['sample_freq'] == 0:
                    self.sample(os.path.join(self.log_root, 'samples', f'epoch_{ep}.png'))
                    sample_paths.append(os.path.join(self.log_root, 'samples', f'epoch_{ep}.png'))

        if self.is_master:
            self.save_model(os.path.join(self.log_root, 'model.pt'))
            self.generate_gif(sample_paths, os.path.join(self.log_root, f'samples.gif'))
            self.writer.close()

    def train_one_epoch(self, ep):
        self.model.train()

        pbar = tqdm(self.train_loader, desc=f'Epoch {ep}', ncols=120) if self.is_master else self.train_loader
        for it, X in enumerate(pbar):
            if isinstance(X, (tuple, list)):
                X = X[0]
            X = X.to(device=self.device, dtype=torch.float32)
            recX, mean, logvar = self.model(X)
            loss_rec = F.mse_loss(recX, X)
            loss_kl = torch.mean(torch.sum(mean ** 2 + torch.exp(logvar) - logvar - 1, dim=1))
            loss = loss_rec + self.config['weight_kl'] * loss_kl

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.is_ddp:
                loss = reduce_tensor(loss.detach(), self.world_size)
                loss_rec = reduce_tensor(loss_rec.detach(), self.world_size)
                loss_kl = reduce_tensor(loss_kl.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('Train/loss', loss.item(), it + ep * len(self.train_loader))
                self.writer.add_scalar('Train/loss_rec', loss_rec.item(), it + ep * len(self.train_loader))
                self.writer.add_scalar('Train/loss_kl', loss_kl.item(), it + ep * len(self.train_loader))
                pbar.set_postfix({'loss': loss.item()})

        if self.is_master:
            pbar.close()

    @torch.no_grad()
    def sample(self, savepath):
        model = self.model.module if self.is_ddp else self.model
        model.eval()
        X = model.generate(self.sample_z).cpu()
        save_image(X, savepath, nrow=8, normalize=True, value_range=(-1, 1))

    @staticmethod
    def generate_gif(img_paths, savepath, duration=0.1):
        images = [imageio.imread(p) for p in img_paths]
        imageio.mimsave(savepath, images, 'GIF', duration=duration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config.yml', help='path to training configuration file')
    args = parser.parse_args()

    trainer = Trainer(args.config_path)
    trainer.train()


if __name__ == '__main__':
    main()
