import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from pytorch_lightning.loggers import WandbLogger

from torchvision import transforms

import pandas as pd
import os

from dataset import CellDataset
from model.counting_model import CSRNet
from model.twobranch_push import countXplain

# from helpers import *
import matplotlib.pyplot as plt
import wandb

import argparse

import time

pl.seed_everything(42)


def train(args):
    """
    Train the model with the given hyperparameters
    """

    num_prototypes = args.num_prototypes
    lr = args.lr
    dataset = args.dataset.split("_")[0]
    fg_coef = args.fg_coef
    diversity_coef = args.diversity_coef
    proto_to_feature_coef = args.proto_to_feature_coef
    batch_size = args.batch_size

    hparams = {
        "num_prototypes": num_prototypes,
        "lr": lr,
        "dataset": dataset,
        "fg_coef": fg_coef,
        "diversity_coef": diversity_coef,
        "proto_to_feature_coef": proto_to_feature_coef,
        "epsilon": 1e-4,  # Distance to similarity conversion
        "train_dir": f"../Datasets/{args.dataset}/trainval/images",
        "val_dir": f"../Datasets/{args.dataset}/test/images",
        "test_dir": f"../Datasets/{args.dataset}/test/images",
        "batch_size": batch_size,
        "num_workers": 4,
    }

   
    ckpt_path = f"checkpoints/{args.dataset}_best.ckpt"

    # Create the model
    ct_model = CSRNet().load_from_checkpoint(ckpt_path)
    # ct_model = CSRNet()
    print("Loaded the pretrained CSRNet model")

    proto_model = countXplain(hparams=hparams, count_model=ct_model)
    print("Created the ProtoModel")

    # prepare data
    proto_model.prepare_data()

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=500,
        check_finite=True,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_counting_loss",
        dirpath=f"reb_outputs/{dataset}_{num_prototypes}/",
        filename=f"{dataset}_{num_prototypes}_div_{diversity_coef}_proto_{proto_to_feature_coef}"+"_{val_counting_loss:.2f}_mae",
        save_top_k=1,
        mode="min",
    )

    wandb_logger = WandbLogger(
        project=f"MIDL - {dataset} - REBUTTAL-FINAL", name=f"push_{dataset}_64_{num_prototypes}_{lr}"
    )

    class UnfreezeCallback(pl.Callback):
        def __init__(self, unfreeze_epoch=50):
            super().__init__()
            self.unfreeze_epoch = unfreeze_epoch

        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch == self.unfreeze_epoch:
                # pl_module.hparams['lr'] = 1e-7
                for param in pl_module.front_end.parameters():
                    param.requires_grad = True
                for param in pl_module.back_end.parameters():
                    param.requires_grad = True
                print("Unfreezing the pretrained model")

    class SimilaritySampler(pl.Callback):
        """
        After every 10 epochs, chooses a random image and perform a forward pass to get the similarity maps for each prototype
        """
        def __init__(self):
            super().__init__()

        def on_train_epoch_end(self, trainer, pl_module):

            if trainer.current_epoch % 10 == 0:
                # choose a random image
                dataset = pl_module.train_dataset
                idx = torch.randint(0, len(dataset), (1,)).item()
                img, _ = dataset[idx]

                num_prototypes = pl_module.prototypes.shape[0]

                img = img.to(pl_module.device)

                # forward pass
                with torch.no_grad():
                    _, _, distances = pl_module(img.unsqueeze(0))

                    # Convert distances to similarity
                    similarities = pl_module.distance2similarity(distances)

                    # Create a figure with the original image, and the similarity maps. There is num_prototypes similarity maps
                    fig, axs = plt.subplots(1, num_prototypes + 1, figsize=(20, 5))
                    axs[0].imshow(img.squeeze().permute(1, 2, 0).cpu().numpy())
                    axs[0].set_title("Original Image")
                    for i in range(num_prototypes):
                        axs[i + 1].imshow(
                            similarities[0, i].squeeze().cpu().numpy(), cmap="jet"
                        )
                        axs[i + 1].set_title(f"Similarity Map {i + 1}")

                    wandb_logger.experiment.log(
                        {
                            "Similarity Maps": wandb.Image(fig),
                        }
                    )

                    plt.close(fig)

    trainer = pl.Trainer(
        accelerator="gpu",
        gpus = 1,
        max_epochs=500,
        # accumulate_grad_batches=8,
        callbacks=[
            RichProgressBar(),
            checkpoint_callback,
            early_stop_callback,
            # UnfreezeCallback(unfreeze_epoch=80),
            # SimilaritySampler(),
        ],
        logger=wandb_logger,
        # detect_anomaly=True,
    )

    # log the hyperparameters
    trainer.logger.log_hyperparams(hparams)
    # add tag to the experiment
    wandb_logger.experiment.tags = [args.dataset]

    # fit model
    trainer.fit(proto_model)

    # Load the best model
    proto_model = countXplain(hparams, ct_model).load_from_checkpoint(
        checkpoint_callback.best_model_path, count_model=ct_model
    )

    # Test the model
    test_results = trainer.test(proto_model)

    # # load results csv file if exists and append new results
    # if os.path.exists("results_unfreeze.csv"):
    #     # create a row with the hyperparameters and the test results
    #     row = hparams
    #     row.update(test_results[0])

    #     # append the row to the results csv file
    #     results = pd.read_csv("results_unfreeze.csv")
    #     results = pd.concat([results, pd.DataFrame(row, index=[0])])

    #     results.to_csv("results_midl.csv", index=False)
    # else:
    #     # create a row with the hyperparameters and the test results
    #     row = hparams
    #     row.update(test_results[0])

    #     # create a new results csv file
    #     results = pd.DataFrame(row, index=[0])
    #     results.to_csv("results_unfreeze.csv", index=False)

    # close the wandb logger
    wandb_logger.experiment.finish()

    return trainer.callback_metrics["test_loss"].item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DCC_2")
    parser.add_argument("--num_prototypes", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--diversity_coef", type=float, default=100)
    parser.add_argument("--proto_to_feature_coef", type=float, default=1)
    parser.add_argument("--fg_coef", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=32)

    return parser.parse_args()


def main():
    args = parse_args()

    train(args)

    

if __name__ == "__main__":
    main()
