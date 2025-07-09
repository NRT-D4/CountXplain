import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from glob import glob

from PIL import Image
import cv2
import h5py

from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

from rich.progress import track

import pytorch_lightning as pl

class CellDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # self.all_images = glob(os.path.join(self.root_dir, "*.tiff"))
        self.all_images = glob(os.path.join(self.root_dir, "*.png"))

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        image_path = self.all_images[idx]
        image = np.array(Image.open(image_path).convert('RGB'))

        # density_map = h5py.File(image_path.replace(".tiff", ".h5").replace("images", "densities"), 'r')['density']
        density_map = h5py.File(image_path.replace(".png", ".h5").replace("images", "densities"), 'r')['density']
        density_map = np.array(density_map)
        density_map = cv2.resize(density_map, (image.shape[1]//8, image.shape[0]//8),interpolation = cv2.INTER_AREA) * 64
        density_map = torch.tensor(density_map, dtype=torch.float32)

        # Check if there are any NaNs in the density map. Print the image path if there are
        if torch.isnan(density_map).any():
            print(image_path)
            print(f"Cell count: {density_map.sum()}")

        if self.transform:
            image = self.transform(image)

        

        return image, density_map.unsqueeze(0)
    


class CellDatasetFCRN(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # self.all_images = glob(os.path.join(self.root_dir, "*.tiff"))
        self.all_images = glob(os.path.join(self.root_dir, "*.png"))

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        image_path = self.all_images[idx]
        image = np.array(Image.open(image_path).convert('RGB'))

        # density_map = h5py.File(image_path.replace(".tiff", ".h5").replace("images", "densities"), 'r')['density']
        density_map = h5py.File(image_path.replace(".png", ".h5").replace("images", "densities"), 'r')['density']
        density_map = np.array(density_map)
        
        
        

        if self.transform:
            # perform horizontal and vertical flipping randomly
            if np.random.rand() > 0.5:
                # print("Flipping horizontally")
                image = np.flip(image, 1).copy()
                density_map = np.flip(density_map, 1).copy()

            if np.random.rand() > 0.5:
                # print("Flipping vertically")
                image = np.flip(image, 0).copy()
                density_map = np.flip(density_map, 0).copy()

        image = transforms.ToTensor()(image)
        density_map = torch.tensor(density_map, dtype=torch.float32) * 100

        # Check if there are any NaNs in the density map. Print the image path if there are
        if torch.isnan(density_map).any():
            print(image_path)
            print(f"Cell count: {density_map.sum()}")

        return image, density_map.unsqueeze(0)
    
