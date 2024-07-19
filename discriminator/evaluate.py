from data_loader import TransformerDataset
from models.vision_language_transformer_sigmoid import VisionLanguageTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import random
import numpy as np

from tokenize_data_2 import PhotoBookDataset
from torch.utils.data import DataLoader

from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def plot_single_img():
    val_dataset = PhotoBookDataset('val')
    THRESHOLD = .8
    PATH = "model.pth"

    device = torch.device('cpu')
    model = VisionLanguageTransformer(N_blocks=6, device=device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # Loop for 100 rows
    for row in range(100):
        fig = plt.figure(constrained_layout=True, figsize=(8, 1))
        subfig = fig.subfigures(1, 1)  # Only 1 row per figure
        indices = random.choices(range(len(val_dataset)), k=1)  # Only 1 sample per figure

        i = indices[0]
        data = val_dataset.data[i]
        images = [Image.open(f'../images/{img}') for img in data['image_paths']]
        label = data['label']
        text = data['raw_text']

        text_embedding, image_embeddings, label = val_dataset[i]
        text_embedding, image_embeddings = text_embedding.to(device), image_embeddings.to(device)

        outputs = model(text_embedding.unsqueeze(0), image_embeddings.unsqueeze(0))

        prediction = outputs.argmax().item()
        confidence = outputs.max().item()

        padding = 20
        axs = subfig.subplots(1, len(images))
        one_wrong = False
        for j, img in enumerate(images):
            if j == label and outputs[j // 6, j % 6] > THRESHOLD:  # Ground truth and model prediction match
                img = ImageOps.expand(img, border=padding, fill=(255, 255, 0))
            elif j == label:  # Image is ground truth
                img = ImageOps.expand(img, border=padding, fill=(0, 255, 0))
            elif outputs[j // 6, j % 6] > THRESHOLD:  # Image is predicted by model
                img = ImageOps.expand(img, border=padding, fill=(255, 0, 0))
                one_wrong = True

            axs[j].imshow(img)
            axs[j].axis('off')

        subfig.suptitle(f"{text}")
        if one_wrong:
            plt.savefig(f"test_row_{row}.png")
        plt.close(fig)


def plot_multiple_img():
    val_dataset = PhotoBookDataset('val')
    THRESHOLD = .8
    PATH = "model.pth"

    device = torch.device('cpu')
    model = VisionLanguageTransformer(N_blocks=6, device=device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # Plot some predictions
    fig = plt.figure(constrained_layout=True, figsize=(8, 6))
    subfigs = fig.subfigures(6, 1)  # Change to 6 rows
    indices = random.choices(range(len(val_dataset)), k=6)  # Change to 6 samples

    for idx in range(6):  # Loop for 6 rows
        i = indices[idx]
        data = val_dataset.data[i]
        images = [Image.open(f'../images/{img}') for img in data['image_paths']]
        label = data['label']
        text = data['raw_text']

        text_embedding, image_embeddings, label = val_dataset[i]
        text_embedding, image_embeddings = text_embedding.to(device), image_embeddings.to(device)

        outputs = model(text_embedding.unsqueeze(0), image_embeddings.unsqueeze(0))

        prediction = outputs.argmax().item()
        confidence = outputs.max().item()

        padding = 20
        axs = subfigs[idx].subplots(1, len(images))
        for j, img in enumerate(images):
            if j == label and outputs[j // 6, j % 6] > THRESHOLD:  # Ground truth and model prediction match
                img = ImageOps.expand(img, border=padding, fill=(255, 255, 0))
            elif j == label:  # Image is ground truth
                img = ImageOps.expand(img, border=padding, fill=(0, 255, 0))
            elif outputs[j // 6, j % 6] > THRESHOLD:  # Image is predicted by model
                img = ImageOps.expand(img, border=padding, fill=(255, 0, 0))

            axs[j].imshow(img)
            axs[j].axis('off')

        subfigs[idx].suptitle(f"{text}")
    plt.savefig("test")
    plt.show()



if __name__ == '__main__':
    plot_single_img()