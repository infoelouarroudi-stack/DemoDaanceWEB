import numpy as np
import cv2
import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms

from VideoSkeleton import VideoSkeleton
from Skeleton import Skeleton
from GenVanillaNN import VideoSkeletonDataset, SkeToImageTransform, GenNNSkeImToImage


# -------------------------
# PatchGAN Critic (WGAN => NO sigmoid)
# Input: image (B,3,64,64)
# Output: patch map (B,1,h,w)
# -------------------------
class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super().__init__()
        self.ngpu = ngpu

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 64->32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 32->16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 16->8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),  # 8->7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # 7->6
        )

    def forward(self, input):
        return self.model(input)


class GenGAN:
    """Generate an image from stickman posture image (GAN booster)."""

    def __init__(self, videoSke, loadFromFile=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.netG = GenNNSkeImToImage().to(self.device)
        self.netD = Discriminator().to(self.device)

        self.filename = 'data/Dance/DanceGenGAN.pth'

        image_size = 64

        src_transform = transforms.Compose([
            SkeToImageTransform(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        tgt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Maintenant le dataset renvoie (stickman_img, real_img)
        self.dataset = VideoSkeletonDataset(
            videoSke,
            ske_reduced=True,
            source_transform=src_transform,
            target_transform=tgt_transform
        )
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)

        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            checkpoint = torch.load(self.filename, map_location=self.device)
            if isinstance(checkpoint, dict) and "netG" in checkpoint and "netD" in checkpoint:
                self.netG.load_state_dict(checkpoint["netG"])
                self.netD.load_state_dict(checkpoint["netD"])
            else:
                self.netG = checkpoint.to(self.device)

        self.netG = self.netG.to(self.device)
        self.netD = self.netD.to(self.device)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Compute gradient penalty for WGAN-GP (on images)."""
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        alpha = alpha.expand_as(real_samples)

        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)

        d_interpolated = self.netD(interpolated)  # (B,1,h,w)

        grad_outputs = torch.ones_like(d_interpolated, device=self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, n_epochs=5):
        lambda_gp = 10.0
        n_critic = 5

        lr_g = 1e-4
        lr_d = 4e-4
        optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr_g, betas=(0.0, 0.9))
        optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr_d, betas=(0.0, 0.9))

        l1 = nn.L1Loss()
        lambda_l1 = 100.0

        self.netG.train()
        self.netD.train()

        for epoch in range(n_epochs):
            epoch_d = 0.0
            epoch_g = 0.0

            for stickman_imgs, real_imgs in self.dataloader:
                stickman_imgs = stickman_imgs.to(self.device).float()  # (B,3,64,64) in [-1,1]
                real_imgs = real_imgs.to(self.device).float()          # (B,3,64,64) in [-1,1]

                # ---------------------
                # Train D (n_critic times)
                # ---------------------
                for _ in range(n_critic):
                    optimizer_D.zero_grad()

                    with torch.no_grad():
                        fake_imgs = self.netG(stickman_imgs)  # (B,3,64,64) in [-1,1] si tanh

                    real_validity = self.netD(real_imgs)
                    fake_validity = self.netD(fake_imgs.detach())

                    d_loss = fake_validity.mean() - real_validity.mean()
                    gp = self.compute_gradient_penalty(real_imgs.data, fake_imgs.data)
                    d_loss_total = d_loss + lambda_gp * gp

                    d_loss_total.backward()
                    optimizer_D.step()

                # ---------------------
                # Train G (once)
                # ---------------------
                optimizer_G.zero_grad()

                fake_imgs = self.netG(stickman_imgs)
                fake_validity = self.netD(fake_imgs)

                g_adv = -fake_validity.mean()
                g_l1 = l1(fake_imgs, real_imgs)
                g_loss = g_adv + lambda_l1 * g_l1

                g_loss.backward()
                optimizer_G.step()

                epoch_d += d_loss_total.item()
                epoch_g += g_loss.item()

            print(f"[{epoch+1}/{n_epochs}] D={epoch_d/len(self.dataloader):.4f}  G={epoch_g/len(self.dataloader):.4f}")

        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save({"netG": self.netG.state_dict(), "netD": self.netD.state_dict()}, self.filename)
        print("GenGAN: Saved checkpoint to", self.filename)

    def generate(self, ske):
        """Generate an image from skeleton, using stickman image as input."""
        self.netG.eval()

        src_transform = transforms.Compose([
            SkeToImageTransform(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        stick = src_transform(ske).unsqueeze(0).to(self.device).float()  # (1,3,64,64)

        with torch.no_grad():
            normalized_output = self.netG(stick)[0].detach().cpu()  # (3,64,64) in [-1,1]

        # Réutilise tensor2image du dataset (dénormalise + BGR)
        res = self.dataset.tensor2image(normalized_output)
        return res


if __name__ == '__main__':
    train = True  # Set to False to only test
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "data/taichi1.mp4"

    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train:
        gen = GenGAN(targetVideoSke, loadFromFile=False)
        gen.train(50)
        print("Training completed and model saved. Exiting...")
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)
        
        # Test - only show images if not training
        for i in range(targetVideoSke.skeCount()):
            image = gen.generate(targetVideoSke.ske[i])
            image = cv2.resize(image, (256, 256))
            cv2.imshow('Image', image)
            key = cv2.waitKey(10)
            if key & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
