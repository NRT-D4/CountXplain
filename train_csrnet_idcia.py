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
from model.counting_model import CSRNet

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
        "model_name": "csrnet",
        "batch_size": 1,
        "lr": 1e-6,
        "epochs": 200,
        "dataset": "IDCIA_v1"
    }


def prepare_data(config):

    
    train_dir = f"../Datasets/{config['dataset']}/trainval/images/"
    test_dir = f"../Datasets/{config['dataset']}/test/images/"
    val_dir = test_dir

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_dataset = CellDataset(train_dir, transform=train_transforms)
    val_dataset = CellDataset(val_dir, transform=test_transforms)
    test_dataset = CellDataset(test_dir, transform=test_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def train_model(config):
    wandb.init(project='MIDL - CSRNet_IDCIA_v1')

    # Prepare the data
    train_loader, val_loader, test_loader = prepare_data(config)

    # Model Initialization
    model = CSRNet(learning_rate=config["lr"])

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        dirpath='checkpoints',
        filename=config["dataset"] + '_best',
        save_top_k=1,
        mode='min',
    )

    early_stopping = EarlyStopping('val_mae', patience=150)

    # Logger
    wandb_logger = WandbLogger(project='MIDL - CSRNet_IDCIA_v1', config=config)

    # Training
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator='gpu',
        gpus = 1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=wandb_logger
    )

    trainer.logger.log_hyperparams(config)
    # add tag to the experiment
    wandb_logger.experiment.tags = [args.dataset]

    trainer.fit(model, train_loader, val_loader)

    # Testing
    trainer.test(dataloaders=test_loader, ckpt_path='best')

    wandb.finish()


if __name__ == "__main__":

    config = get_default_config()

   
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="csrnet")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dataset", type=str, default="DCC")

    args = parser.parse_args()

    config.update(vars(args))

    train_model(config)

