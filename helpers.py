import numpy as np
from scipy.ndimage import label
from scipy.stats import wilcoxon

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from torchvision import transforms

from dataset import CellDataset

from model.counting_model import CSRNet
from model.twobranch_push import countXplain
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random
from glob import glob
import h5py
from tqdm import tqdm

import pandas as pd
from scipy.stats import binom_test

from collections import defaultdict

# Helper methods to be used when performing qualitative and quantitative analysis on the prototype model.


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


def find_multiple_high_activation_crops(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = activation_map >= threshold

    labeled_array, num_features = label(mask)

    bounding_boxes = []
    for feature in range(1, num_features + 1):
        locs = np.where(labeled_array == feature)
        y_min, y_max = locs[0].min(), locs[0].max()
        x_min, x_max = locs[1].min(), locs[1].max()
        bounding_boxes.append((y_min, y_max + 1, x_min, x_max + 1))

    return bounding_boxes


# Initializes the proto model and the counting model
def initialize_model(path_to_ckpt, path_to_ct):


    hparams = {
        "num_prototypes": 20,
        "lr": 1e-3,
        "bg_coef": 1,
        "fg_coef": 1,
        "diversity_coef": 1,
        "proto_to_feature_coef": 1,
        "data_coverage_coef": 1,
        "inter_class_coef": 0,
        "epsilon": 1e-4,  # Distance to similarity conversion
        "train_dir": f"../Datasets/DCC/trainval/images",
        "val_dir": f"../Datasets/DCC/test/images",
        "test_dir": f"../Datasets/DCC/test/images",
        "batch_size": 1,
        "num_workers": 4,
        
    }
    ct_model = CSRNet().load_from_checkpoint(path_to_ct)
    proto_model = countXplain(hparams, ct_model).load_from_checkpoint(
        path_to_ckpt, count_model=ct_model
    )
    proto_model.eval()
    proto_model.hparams.batch_size = 1
    proto_model.prepare_data()

    return proto_model


#  Visualizes patches close to each prototype : closest_points(proto_model,similarities,image)
def closest_points(proto_model, similarities, image):
    # Find the patch with the min distance for each prototype
    min_distance, min_pos = torch.max(
        similarities.view(similarities.shape[0], similarities.shape[1], -1), dim=2
    )

    min_pos_2d = torch.stack(
        (min_pos // similarities.shape[3], min_pos % similarities.shape[3]), dim=2
    )

    # show the image
    plt.imshow(image)
    plt.axis("off")

    # Get the scale factor for the patches
    scale_factor_h = 8
    scale_factor_w = 8

    # plot each prototype as a rectangle on the original image with random colors
    for i in range(proto_model.hparams.num_prototypes):
        edge_clr = np.random.rand(
            3,
        )

        plt.gca().add_patch(
            plt.Rectangle(
                (
                    min_pos_2d[0, i, 1] * scale_factor_w + 2,
                    min_pos_2d[0, i, 0] * scale_factor_h + 2,
                ),
                16,
                16,
                linewidth=1,
                edgecolor=edge_clr,
                facecolor="none",
                label=f"Prototype {i}",
            )
        )
        plt.text(
            min_pos_2d[0, i, 1] * scale_factor_w,
            min_pos_2d[0, i, 0] * scale_factor_h,
            f"P{i}",
            fontsize=8,
            color=edge_clr,
        )

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.show()


# Visualizes the similarity map for each prototype: similarity_map(proto_model,similarities)
def similarity_map(proto_model, similarities):
    fig, axs = plt.subplots(2, proto_model.hparams.num_prototypes // 2, figsize=(12, 3))
    # fig.suptitle("Similarity Maps for each Prototype")
    axs = axs.flatten()
    for i in range(proto_model.hparams.num_prototypes):
        sim = similarities[0, i, :, :].detach().numpy()
        # upscale the similarity map
        sim = cv2.resize(
            sim,
            (
                sim.shape[1] * 8,
                sim.shape[0] * 8,
            ),
            interpolation=cv2.INTER_CUBIC,
        ) / 255

        im = axs[i].imshow(sim, cmap="jet")
        # axs[i].set_title(f"Prototype {i}", fontsize=6)
        axs[i].axis("off")
    plt.tight_layout()


    # add a colorbar to the right of the figure
    fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.05, pad=0.05)
    



    # Save the figure for latex report
    plt.savefig("Figures/similarity_map.png", bbox_inches="tight", dpi=400)

    plt.show()


# Shows the prototypes' similarity maps on the original image: overlay_similarity(proto_model,image,similarities)
def overlay_similarity(proto_model, image, similarities):
    fig, axs = plt.subplots(2, proto_model.hparams.num_prototypes // 2, figsize=(10,8))
    # fig.suptitle("Overlayed Similarity Maps for each Prototype")
    axs = axs.flatten()
    for i in range(proto_model.hparams.num_prototypes):
        axs[i].imshow(image)
        sim = similarities[0, i, :, :].detach().numpy() * 255

        if i < proto_model.hparams.num_prototypes // 2:
            thresh = np.percentile(sim, 99)
            sim[sim < thresh] = 0

        sim = cv2.resize(sim, (image.shape[1], image.shape[0]))
        im = axs[i].imshow(sim, cmap="jet", alpha=0.25)
        axs[i].axis("off")
        axs[i].set_title(f"Prototype {i}", fontsize=6)

    plt.tight_layout()

    fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.05, pad=0.05)
    plt.savefig("Figures/overlay_similarity.png", bbox_inches="tight", dpi=400)
    plt.show()


# Plots highly activated patches for each prototype: highly_activated_patches(proto_model,similarities,image,percentile=99.9)
def highly_activated_patches(proto_model, similarities, image, percentile=99.9):
    fig, axs = plt.subplots(2, proto_model.hparams.num_prototypes // 2, figsize=(12, 6))
    fig.suptitle("Highly Activated Patches for each Prototype")
    axs = axs.flatten()

    for i in range(proto_model.hparams.num_prototypes):
        coord = find_high_activation_crop(
            similarities[0, i, :, :].detach().numpy(), percentile
        )

        # translate the coordinates to the original image
        scale_factor_h = 8
        scale_factor_w = 8

        axs[i].imshow(image)
        axs[i].axis("off")

        # plot the rectangle
        axs[i].add_patch(
            plt.Rectangle(
                (coord[2] * scale_factor_w, coord[0] * scale_factor_h),
                (coord[3] - coord[2]) * scale_factor_w,
                (coord[1] - coord[0]) * scale_factor_h,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )

        axs[i].set_title(f"Prototype {i}")
    plt.tight_layout()
    plt.show()


# Crops the highly activated patches for each prototype: crop_highly_activated(proto_model,similarities,image,percentile=99.9)
def crop_highly_activated(proto_model, similarities, image, percentile=99.9):
    fig, axs = plt.subplots(2, proto_model.hparams.num_prototypes // 2, figsize=(12, 4))
    fig.suptitle("Cropped Highly Activated Patches for each Prototype")
    axs = axs.flatten()

    for i in range(proto_model.hparams.num_prototypes):
        coord = find_high_activation_crop(
            similarities[0, i, :, :].detach().numpy(), percentile
        )

        # translate the coordinates to the original image
        scale_factor_h = 8
        scale_factor_w = 8

        coord = [
            coord[0] * scale_factor_h,
            coord[1] * scale_factor_h,
            coord[2] * scale_factor_w,
            coord[3] * scale_factor_w,
        ]

        axs[i].imshow(image[coord[0] : coord[1], coord[2] : coord[3]])
        axs[i].axis("off")
        axs[i].set_title(f"Prototype {i}")
    plt.tight_layout()
    plt.show()


# Crops closest patches for each prototype: crop_closest(proto_model,similarities,image)
def crop_closest(proto_model, similarities, image):
    # Find the patch with the min distance for each prototype
    min_distance, min_pos = torch.max(
        similarities.view(similarities.shape[0], similarities.shape[1], -1), dim=2
    )

    min_pos_2d = torch.stack(
        (min_pos // similarities.shape[3], min_pos % similarities.shape[3]), dim=2
    )

    # Get the scale factor for the patches
    scale_factor_h = 8
    scale_factor_w = 8

    fig, axs = plt.subplots(2, proto_model.hparams.num_prototypes // 2, figsize=(12, 4))
    fig.suptitle("Cropped Closest Patches for each Prototype")
    axs = axs.flatten()

    # plot each prototype as a rectangle on the original image with random colors
    for i in range(proto_model.hparams.num_prototypes):
        coord = [
            min_pos_2d[0, i, 0] * scale_factor_h,
            min_pos_2d[0, i, 0] * scale_factor_h + 16,
            min_pos_2d[0, i, 1] * scale_factor_w,
            min_pos_2d[0, i, 1] * scale_factor_w + 16,
        ]

        axs[i].imshow(image[coord[0] : coord[1], coord[2] : coord[3]])
        axs[i].axis("off")
        axs[i].set_title(f"Prototype {i}")
    plt.tight_layout()
    plt.show()


# Visualizes multiple highly activated patches for each prototype: visualize_highly_activated(proto_model,similarities,image,percentile=99.9)
def multiple_highly_activated(proto_model, similarities, image, percentile=99.9):
    fig, axs = plt.subplots(2, proto_model.hparams.num_prototypes // 2, figsize=(12, 6))
    fig.suptitle("Multiple Highly Activated Patches for each Prototype")
    axs = axs.flatten()

    for i in range(proto_model.hparams.num_prototypes):
        boxes = find_multiple_high_activation_crops(
            similarities[0, i, :, :].detach().numpy(), percentile
        )
        scale_factor_h = 8
        scale_factor_w = 8

        axs[i].imshow(image)
        axs[i].axis("off")

        for box in boxes:
            axs[i].add_patch(
                plt.Rectangle(
                    (box[2] * scale_factor_w, box[0] * scale_factor_h),
                    (box[3] - box[2]) * scale_factor_w,
                    (box[1] - box[0]) * scale_factor_h,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
            )

        axs[i].set_title(f"Prototype {i}")
    plt.tight_layout()
    plt.show()



# Visualizes multiple highly activated patches for each prototype: visualize_highly_activated(proto_model,similarities,image,percentile=99.9)
def multiple_highly_activated_exp(proto_model, similarities, image, percentile=99.9):
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    
    
    for i in range(proto_model.hparams.num_prototypes):
        fig, axs = plt.figure(figsize=(4, 3))
        boxes = find_multiple_high_activation_crops(
            similarities[0, i, :, :].detach().numpy(), percentile
        )
        scale_factor_h = 8
        scale_factor_w = 8

        plt.imshow(image)
        plt.axis("off")

        for box in boxes:
            plt.add_patch(
                plt.Rectangle(
                    (box[2] * scale_factor_w, box[0] * scale_factor_h),
                    (box[3] - box[2]) * scale_factor_w,
                    (box[1] - box[0]) * scale_factor_h,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
            )
        plt.show()
        fig.close()
        # axs[i].set_title(f"Prototype {i}")
    # plt.tight_layout()
    # plt.show()

# Crops multiple highly activated patches for each prototype: crop_multiple_highly_activated(proto_model,similarities,image,percentile=99.9)
def crop_multiple_highly_activated(proto_model, similarities, image, percentile=99.9):
    # Create a grid inside a grid to show the cropped patches of each prototype

    for i in range(proto_model.hparams.num_prototypes):
        boxes = find_multiple_high_activation_crops(
            similarities[0, i, :, :].detach().numpy(), percentile
        )
        scale_factor_h = 8
        scale_factor_w = 8

        # check if there are more than 1 boxes
        if len(boxes) > 1:
            fig, axs = plt.subplots(1, len(boxes), figsize=(len(boxes) * 1, 3))
            fig.suptitle(f"Prototype {i}")
            axs = axs.flatten()
            for j in range(len(boxes)):
                coord = [
                    boxes[j][0] * scale_factor_h,
                    boxes[j][1] * scale_factor_h,
                    boxes[j][2] * scale_factor_w,
                    boxes[j][3] * scale_factor_w,
                ]
                axs[j].imshow(image[coord[0] : coord[1], coord[2] : coord[3]])
                axs[j].axis("off")
                axs[j].set_title(f"Patch {j}")
            plt.tight_layout()
            # save the image for the latex report in a folder for each prototype
            # plt.savefig(f"Figures/global/prototype_{i}/{random.randint(10000,20000)}.png", bbox_inches="tight", dpi=400)
            # plt.show()
        else:
            coord = [
                boxes[0][0] * scale_factor_h,
                boxes[0][1] * scale_factor_h,
                boxes[0][2] * scale_factor_w,
                boxes[0][3] * scale_factor_w,
            ]
            plt.imshow(image[coord[0] : coord[1], coord[2] : coord[3]])
            plt.axis("off")
            plt.title(f"Prototype {i}")
            plt.tight_layout()
            # save the image for the latex report in a folder for each prototype
            # plt.savefig(f"Figures/global/prototype_{i}/{random.randint(10000,20000)}.png", bbox_inches="tight", dpi=400)
            # plt.show()


# Get a random image from the dataset
def get_random_image(dataset="VGG"):
    if dataset == "VGG":
        all_imgs = glob("../Datasets/VGG/trainval/images/*.png")
    elif dataset == "DCC":
        all_imgs = glob("../Datasets/DCC/trainval/images/*.png")
    elif dataset == "IDCIA_v1":
        all_imgs = glob("../Datasets/IDCIA_v1/trainval/images/*.png")

    img_path = random.choice(all_imgs)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    density_map = h5py.File(
        img_path.replace(".png", ".h5").replace("images", "densities"), "r"
    )["density"]
    density_map = np.array(density_map)

    return img, density_map


# Apply transforms on the image
def apply_transforms(img):
    tr = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = tr(img)

    return img


# A function to reverse normalize the image: reverse_normalize(image)
def reverse_normalize(image):
    img = image[0, :, :, :].detach().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    return img


# Need to add function to mask(erase) closeset patch to prototype and how it affects the count.


# Erase the closest patch to each prototype: erase_closest_patch(proto_model,similarities,image)
def erase_closest_patch(proto_model, similarities, image):
    erased_images = []

    for i in range(proto_model.hparams.num_prototypes):
        # First find multiple highly activated patches
        boxes = find_multiple_high_activation_crops(
            similarities[0, i, :, :].detach().numpy(), 99
        )
        scale_factor_h = 8
        scale_factor_w = 8

        # create a copy of the image
        img = image.copy()

        for box in boxes:
            coord = [
                box[0] * scale_factor_h,
                box[1] * scale_factor_h,
                box[2] * scale_factor_w,
                box[3] * scale_factor_w,
            ]
            img[coord[0] : coord[1], coord[2] : coord[3]] = np.mean(img)

        erased_images.append(img)

    return erased_images


# A function that takes a list of images, a prototype model and a counting model. For each image, it first makes a prediction using the prototype model and the counting model. Then it erases the closest patch to each prototype and makes a prediction using the counting model and the prototype model. For each prototype, if the count decreases at least by 1, it means that the patch was important for the count. If the count doesn't decrease, it means that the patch was not important for the count.
def proto_faithfulness(images, proto_model, count_model):
    assert len(images) > 0, "The images list should not be empty"

    # get the number of prototypes
    num_prototypes = proto_model.hparams.num_prototypes // 2

    # create a dictionary to store the results for each prototype. Initialize the dictionary as {prototype 1: 0, prototype 2: 0, ...}
    prototype_results = dict(zip(range(num_prototypes), [0] * num_prototypes))

    # iterate over the images. the images list should contain paths to the images
    for img_path in images:
        # read the image
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print(f"Error opening image {img_path}")
            continue

        img = np.array(img)

        # read the density map
        density_map = h5py.File(
            img_path.replace(".png", ".h5").replace("images", "densities"), "r"
        )["density"]
        density_map = np.array(density_map)

        # apply transforms
        image = apply_transforms(img)

        # make a prediction using the prototype model
        fmaps, fg, distances = proto_model(image.unsqueeze(0))
        similarities = proto_model.distance2similarity(distances)

        # make a prediction using the counting model
        count_pred = count_model(image.unsqueeze(0))
        count_pred = count_pred.detach().squeeze(0).squeeze(0).detach().cpu().numpy()
        count_pred = count_pred.sum()

        # erase the closest patch to each prototype
        erased_images = erase_closest_patch(proto_model, similarities, img)

        # create a list to store the counts
        erased_counts = []

        # iterate over the erased images and make a prediction using the counting model
        for erased_img in erased_images:
            erased_img = apply_transforms(erased_img)
            erased_count = (
                count_model(erased_img.unsqueeze(0))
                .squeeze(0)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            erased_count = erased_count.sum()
            erased_counts.append(erased_count)

        # itrate over the counts and compare them with the original count, If the count decreases at least by 1, increase the counter by 1.
        for i in range(num_prototypes):
            if erased_counts[i] <= count_pred - 1:
                prototype_results[i] += 1

    return prototype_results


# A function that calculates wilcoxon signed rank test to show if the prototypes are significantly important for the count.
def wilcoxon_signed_rank_test(images,proto_model,count_model):
    assert len(images) > 0, "The images list should not be empty"

    num_prototypes = proto_model.hparams.num_prototypes

    print(f"Calculating the Wilcoxon signed rank test for {num_prototypes} prototypes")

    # create dictionary to store lists of counts for each prototype
    prototype_results = defaultdict(list)

    original_counts = []

    for img_path in tqdm(images):
        # read the image
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print(f"Error opening image {img_path}")
            continue

        img = np.array(img)

        # read the density map
        density_map = h5py.File(
            img_path.replace(".png", ".h5").replace("images", "densities"), "r"
        )["density"]
        density_map = np.array(density_map)

        # apply transforms
        image = apply_transforms(img)

        # make a prediction using the prototype model
        fmaps, fg, distances = proto_model(image.unsqueeze(0))
        similarities = proto_model.distance2similarity(distances)

        # make a prediction using the counting model
        count_pred = count_model(image.unsqueeze(0))
        count_pred = count_pred.detach().squeeze(0).squeeze(0).detach().cpu().numpy()
        count_pred = count_pred.sum()

        # erase the closest patch to each prototype
        erased_images = erase_closest_patch(proto_model, similarities, img)

        erased_counts = []

        for erased_img in erased_images:
            erased_img = apply_transforms(erased_img)
            erased_count = (
                count_model(erased_img.unsqueeze(0))
                .squeeze(0)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            erased_count = erased_count.sum()
            erased_counts.append(erased_count)

        for i in range(num_prototypes):
            prototype_results[i].append(erased_counts[i])

        original_counts.append(count_pred)



    # perform the wilcoxon signed rank test
    p_values = []
    for i in range(num_prototypes):
        p_value = wilcoxon(original_counts, prototype_results[i],alternative="greater").pvalue
        p_values.append(p_value)

    return p_values

def mask_high_activation_pixels(similarities,image, percentile=95):
    erased_images = []

    for i in range(similarities.shape[1]):

        sim = cv2.resize(
            similarities[0, i, :, :].detach().numpy(),
            (
                image.shape[1],
                image.shape[0],
            ),
        )

        threshold = np.percentile(sim, percentile)
        mask = sim >= threshold

        img = image.copy()

        img[mask] = 0

        erased_images.append(img)

    return erased_images


def statistical_test(images, proto_model, count_model):
    assert len(images) > 0, "The images list should not be empty"

    num_prototypes = proto_model.hparams.num_prototypes

    print(f"Calculating the Wilcoxon signed rank test for {num_prototypes} prototypes")

    # create dictionary to store lists of counts for each prototype
    prototype_results = defaultdict(list)

    original_counts = []

    for img_path in tqdm(images):
        # read the image
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print(f"Error opening image {img_path}")
            continue

        img = np.array(img)

        # read the density map
        density_map = h5py.File(
            img_path.replace(".png", ".h5").replace("images", "densities"), "r"
        )["density"]
        density_map = np.array(density_map)

        # apply transforms
        image = apply_transforms(img)

        # make a prediction using the prototype model
        fmaps, fg, distances = proto_model(image.unsqueeze(0))
        similarities = proto_model.distance2similarity(distances)

        # make a prediction using the counting model
        count_pred = count_model(image.unsqueeze(0))
        count_pred = count_pred.detach().squeeze(0).squeeze(0).detach().cpu().numpy()
        count_pred = count_pred.sum()

        # erase the closest patch to each prototype
        erased_images = mask_high_activation_pixels(similarities, img)

        erased_counts = []

        for erased_img in erased_images:
            erased_img = apply_transforms(erased_img)
            erased_count = (
                count_model(erased_img.unsqueeze(0))
                .squeeze(0)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            erased_count = erased_count.sum()
            erased_counts.append(erased_count)

        for i in range(num_prototypes):
            prototype_results[i].append(erased_counts[i])

        original_counts.append(count_pred)

    # save the original counts and the prototype results as csv files
    df = pd.DataFrame(original_counts, columns=["Original Count"])
    for i in range(num_prototypes):
        df[f"Prototype {i}"] = prototype_results[i]

    df.to_csv("results.csv", index=False)

    # perform the wilcoxon signed rank test
    p_values = []
    for i in range(num_prototypes):
        p_value = wilcoxon(original_counts, prototype_results[i], alternative="greater").pvalue
        p_values.append(p_value)

    return p_values


def sign_test(images,proto_model,count_model):
    assert len(images) > 0, "The images list should not be empty"

    num_prototypes = proto_model.hparams.num_prototypes

    print(f"Calculating the Wilcoxon signed rank test for {num_prototypes} prototypes")

    # create dictionary to store lists of counts for each prototype
    prototype_results = defaultdict(list)

    original_counts = []

    for img_path in tqdm(images):
        # read the image
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print(f"Error opening image {img_path}")
            continue

        img = np.array(img)

        # read the density map
        density_map = h5py.File(
            img_path.replace(".png", ".h5").replace("images", "densities"), "r"
        )["density"]
        density_map = np.array(density_map)

        # apply transforms
        image = apply_transforms(img)

        # make a prediction using the prototype model
        fmaps, fg, distances = proto_model(image.unsqueeze(0))
        similarities = proto_model.distance2similarity(distances)

        # make a prediction using the counting model
        count_pred = count_model(image.unsqueeze(0))
        count_pred = count_pred.detach().squeeze(0).squeeze(0).detach().cpu().numpy()
        count_pred = count_pred.sum()

        # erase the closest patch to each prototype
        erased_images = mask_high_activation_pixels(similarities, img)

        erased_counts = []

        for erased_img in erased_images:
            erased_img = apply_transforms(erased_img)
            erased_count = (
                count_model(erased_img.unsqueeze(0))
                .squeeze(0)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            erased_count = erased_count.sum()
            erased_counts.append(erased_count)

        for i in range(num_prototypes):
            prototype_results[i].append(erased_counts[i])

        original_counts.append(count_pred)


    # perform the wilcoxon signed rank test
    p_values = []
    for i in range(num_prototypes):
        # perform a one sided sign test
        # find the differences between the original counts and the prototype results
        differences = np.array(original_counts) - np.array(prototype_results[i])

        negative_differences = np.sum(differences < 0)

        p_value = binom_test(negative_differences, len(differences), 0.5, alternative="less")




        p_values.append(p_value)

    return p_values