import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, EarlyStopping
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import WandbLogger

from torchvision import transforms

import pandas as pd
import os

from dataset import CellDataset, CellDatasetFCRN
from torch.utils.data import DataLoader
from model.counting_model import CSRNet, FCRN_A
from model.twobranch import countXplain

from helpers import *
import matplotlib.pyplot as plt
import wandb

from glob import glob
import argparse

import time

pl.seed_everything(42)

# A wandb sweep to find the best hyperparameters
def get_default_config():
    """Define the default configuration"""
    return {
        "model_name": "fcrn",
        "batch_size": 2,
        "lr": 0.001,
        "epochs": 500,
        "dataset": "DCC"
    }

def prepare_data(config):
    train_dir = f"../Datasets/{config['dataset']}/trainval/images/"
    test_dir = f"../Datasets/{config['dataset']}/test/images/"
    val_dir = test_dir

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CellDatasetFCRN(train_dir, transform=train_transforms)
    val_dataset = CellDatasetFCRN(val_dir, transform=test_transforms)
    test_dataset = CellDatasetFCRN(test_dir, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

def train_model(config):
    wandb.init()
    config = wandb.config

    # Prepare the data
    train_loader, val_loader, test_loader = prepare_data(config)

    # Model Initialization
    model = FCRN_A(config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        dirpath='checkpoints',
        filename='fcrn-{epoch:02d}-{val_mae:.2f}',
        save_top_k=1,
        mode='min',
    )

    early_stopping = EarlyStopping('val_mae', patience=150)

    # Logger
    wandb_logger = WandbLogger()

    # Training
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator='auto',
        callbacks=[checkpoint_callback, early_stopping],
        logger=wandb_logger
    )

    trainer.fit(model, train_loader, val_loader)

    # Testing
    trainer.test(dataloaders=test_loader, ckpt_path='best')

    wandb.finish()

if __name__ == "__main__":
    sweep_config = {
        "method": "random",
        "metric": {
            "name": "val_mae",
            "goal": "minimize"
        },
        "parameters": {
            "learning_rate": {"min": 1e-6, "max": 1e-2},
            "batch_size": {
                "values": [2,8,16]
            },
            "dataset": {
                "values": ["DCC"]
        }
    }
    }


    sweep_id = wandb.sweep(sweep_config, project="fcrn_dcc")

    wandb.agent(sweep_id, function=lambda: train_model(wandb.config), count=20)

 



