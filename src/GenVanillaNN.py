import numpy as np
import cv2
import os
import pickle
import sys
import math


from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image


import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


#from tensorboardX import SummaryWriter


from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton


torch.set_default_dtype(torch.float32)



class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size


    def __call__(self, ske):
        #image = Image.new('RGB', (self.imsize, self.imsize), (255, 255, 255))
        image = white_image = np.zeros((self.imsize, self.imsize, 3), dtype=np.uint8)
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image)
        # key = cv2.waitKey(-1)
        return image




class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """ videoSkeleton dataset:
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ",Skeleton.full_dim,")" )



    def __len__(self):
        return self.videoSke.skeCount()



    def __getitem__(self, idx):
        # prepreocess skeleton (input)
        reduced = True
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        # prepreocess image (output)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image


   
    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            # TP-TODO: Convertir en tenseur
            ske = torch.from_numpy( ske.__array__(reduced=self.ske_reduced).flatten() )
            ske = ske.to(torch.float32)
            # Pas besoin de reshape ici si c'est pour l'approche vecteur (dim 26)
            # Si c'est pour l'approche image, source_transform s'en chargera
        return ske



    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().cpu().numpy()
        # Réorganiser les dimensions (C, H, W) en (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # passage a des images cv2 pour affichage
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denormalized_output = denormalized_image * 1
        return denormalized_output





def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class GenNNSke26ToImage(nn.Module):
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton_dim26)->Image
    """
    def __init__(self):
        super().__init__()
        self.input_dim = Skeleton.reduced_dim # 26
       
        # TP-TODO: Architecture Perceptron Multi-Couches (MLP) -> Reshape -> Convolutions
        # L'idée : On part de 26 valeurs, on les projette vers un espace plus grand (ex: 4*4*256)
        # Puis on utilise des convolutions transposées pour grandir jusqu'à 64x64
       
        self.fc = nn.Linear(self.input_dim, 256 * 4 * 4) # Projection vers 4x4x256
       
        self.model = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(256),
           
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(128),
           
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(64),
           
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(32),
           
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Sortie entre -1 et 1
        )
        print(self.model)


    def forward(self, z):
        # z est de taille (batch, 26) ou (batch, 26, 1, 1)
        z = z.view(z.size(0), -1) # Aplatir en (batch, 26)
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4) # Remettre en forme (batch, 256, 4, 4)
        img = self.model(x)
        return img






class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class GenNNSkeImToImage(nn.Module):
    """ class that Generate a new image from from THE IMAGE OF the new skeleton posture
       SkeletonImage is an image with the skeleton drawed on it
       Fonc generator(SkeletonImage)->Image
    """
    def __init__(self):
        super().__init__()
        # TP-TODO: Architecture U-Net Simplifiée
       
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1) # -> 32x32
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1) # -> 16x16
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1) # -> 8x8
        self.attn = SelfAttention(256)
        self.enc4 = nn.Conv2d(256, 512, 4, 2, 1) # -> 4x4
        
        # Bottleneck with stacked residual blocks
        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )
       
        # Decoder
        self.dec1 = nn.ConvTranspose2d(512, 256, 4, 2, 1) # -> 8x8
        self.dec2 = nn.ConvTranspose2d(512, 128, 4, 2, 1) # -> 16x16
        self.dec3 = nn.ConvTranspose2d(256, 64, 4, 2, 1) # -> 32x32
        self.dec4 = nn.ConvTranspose2d(128, 3, 4, 2, 1) # -> 64x64
       
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()


    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        e3 = self.attn(e3)
        e4 = self.relu(self.enc4(e3))
        
        # Apply bottleneck residual blocks
        bottleneck = self.bottleneck(e4)
       
        # Decoder (avec Skip Connections)
        d1 = self.relu(self.dec1(bottleneck))
        d1 = torch.cat([d1, e3], dim=1)
        
        d2 = self.relu(self.dec2(d1))
        d2 = torch.cat([d2, e2], dim=1)

        d3 = self.relu(self.dec3(d2))
        d3 = torch.cat([d3, e1], dim=1)

        img = self.tanh(self.dec4(d3))
        return img







class GenVanillaNN():
    """ class that Generate a new image from a new skeleton posture
        Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        self.optSkeOrImage = optSkeOrImage # On garde l'option en mémoire
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"GenVanillaNN: device={self.device}")
       
        if optSkeOrImage==1:        # skeleton_dim26 to image
            print("Architecture: GenNNSke26ToImage (Vecteur -> Image)")
            self.netG = GenNNSke26ToImage()
            # Transformation simple : juste conversion en tenseur
            src_transform = None # Sera géré dans preprocessSkeleton
            self.filename = 'data/Dance/DanceGenVanillaFromSke26.pth'
        else:                       # skeleton_image to image
            print("Architecture: GenNNSkeImToImage (Image Squelette -> Image)")
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([ SkeToImageTransform(image_size),
                                                 transforms.ToTensor(),
                                                 # On ne normalise pas l'entrée squelette (0 ou 1 c'est bien)
                                                 ])
            self.filename = 'data/Dance/DanceGenVanillaFromSkeim.pth'


        tgt_transform = transforms.Compose([
                            transforms.Resize((image_size, image_size)),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
                            # ouput image (target) are in the range [-1,1] after normalization
       
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)
       
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            self.netG.load_state_dict(torch.load(self.filename, map_location=self.device))
        
        self.netG.to(self.device)



    def train(self, n_epochs=20):
        # TP-TODO: Boucle d'entraînement standard
        optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.MSELoss()
       
        self.netG.train()
        print(f"Début de l'entraînement pour {n_epochs} epochs...")
       
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, (ske_batch, img_batch) in enumerate(self.dataloader):
                ske_batch, img_batch = ske_batch.to(self.device), img_batch.to(self.device)
               
                optimizer.zero_grad()
               
                # Forward
                output = self.netG(ske_batch)
               
                # Loss
                loss = criterion(output, img_batch)
               
                # Backward
                loss.backward()
                optimizer.step()
               
                running_loss += loss.item()
           
            # Affichage tous les 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{n_epochs}, Loss: {running_loss/len(self.dataloader):.4f}")
               
        # Sauvegarde
        torch.save(self.netG.state_dict(), self.filename)
        print("Modèle sauvegardé :", self.filename)



    def generate(self, ske):
        """ generator of image from skeleton """
        self.netG.eval()
       
        # Préparation du squelette comme dans le dataset
        ske_t = self.dataset.preprocessSkeleton(ske)
       
        # Ajout de la dimension Batch (1, ...)
        if self.optSkeOrImage == 1:
            # Pour Ske26, preprocess renvoie déjà un tenseur, on ajoute juste le batch
            ske_t_batch = ske_t.unsqueeze(0).to(self.device)
        else:
            # Pour SkeIm, preprocess applique la transformation (Image) donc on a (3, 64, 64)
            # On ajoute le batch -> (1, 3, 64, 64)
            ske_t_batch = ske_t.unsqueeze(0).to(self.device)
           
        with torch.no_grad():
            normalized_output = self.netG(ske_t_batch)
           
        # Conversion du résultat en image affichable
        res = self.dataset.tensor2image(normalized_output[0])       # get image 0 from the batch
        return res



if __name__ == '__main__':
    force = False
    optSkeOrImage = 1
    n_epoch = 150
    train = 1 # Met à 1 pour entraîner


    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
       
    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)


    targetVideoSke = VideoSkeleton(filename)


    if train:
        # Train
        # On spécifie optSkeOrImage ici
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False, optSkeOrImage=optSkeOrImage)
        gen.train(n_epoch)
        print("Training completed and model saved. Exiting...")
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True, optSkeOrImage=optSkeOrImage)    # load from file        

        # Test - only show images if not training
        for i in range(targetVideoSke.skeCount()):
            image = gen.generate(targetVideoSke.ske[i])
            # Resize pour mieux voir
            image = cv2.resize(image, (256, 256))
            cv2.imshow('Image', image)
            key = cv2.waitKey(-1)