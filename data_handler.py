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



import pytorch_lightning as pl

class CellDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # self.all_images = glob(os.path.join(self.root_dir, "*.tiff"))
        self.all_images = glob(os.path.join(self.root_dir, "*.jpg"))
        print(f"Found {len(self.all_images)} images in {self.root_dir}")

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        image_path = self.all_images[idx]
        image = np.array(Image.open(image_path).convert('RGB'))

        # density_map = h5py.File(image_path.replace(".tiff", ".h5").replace("images", "densities"), 'r')['density']
        density_map = h5py.File(image_path.replace(".jpg", ".h5").replace("images", "densities"), 'r')['density']
        density_map = np.array(density_map)
        density_map = cv2.resize(density_map, (image.shape[1]//8, image.shape[0]//8),interpolation = cv2.INTER_AREA) * 64
        


        if self.transform:
            image, density_map = self._augmentations(image, density_map)

        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        density_map = torch.tensor(density_map, dtype=torch.float32)
        density_map = density_map.unsqueeze(0)

        return image, density_map
            
    
    def _augmentations(self, image, density_map):
        # perform horizontal and vertical flipping randomly
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1).copy()
            density_map = cv2.flip(density_map, 1).copy()

        if np.random.rand() > 0.5:
            image = np.ascontiguousarray(cv2.flip(image, 0))
            density_map = np.ascontiguousarray(cv2.flip(density_map, 0))

        if np.random.rand() > 0.5:
            # rotate the image by 90 clockwise or anti-clockwise
            k = np.random.randint(1, 4)
            image = np.ascontiguousarray(np.rot90(image, k))
            density_map = np.ascontiguousarray(np.rot90(density_map, k))


        return image, density_map
    


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
    

# train_dataset = CellDataset("../Datasets/DCC_0/trainval/images",transform=True)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

# for i, (image, density_map) in enumerate(train_loader):
#     print(image.shape)
#     print(density_map.shape)
#     break