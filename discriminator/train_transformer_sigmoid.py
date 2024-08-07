from data_loader import TransformerDataset
from models.vision_language_transformer_sigmoid import VisionLanguageTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import random
import numpy as np

from tokenize_data import PhotoBookDataset
from torch.utils.data import DataLoader

from PIL import Image, ImageOps
import matplotlib.pyplot as plt


LEARNING_RATE = 2e-4
BATCH_SIZE = 64 * 4
N_EPOCHS = 250
THRESHOLD = 0.8

LR_GAMMA = 0.99


def plot_samples(num_samples=5):
    dataset = PhotoBookDataset("train")
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    subfigs = fig.subfigures(num_samples, 1)

    indices = random.choices(range(len(dataset)), k=num_samples)

    for i in range(num_samples):
        data = dataset.data[indices[i]]
        images = [Image.open(f"../images/{img}") for img in data["image_paths"]]
        label = data["label"]
        text = data["raw_text"]
        # create a subplot showing all images next to each other
        axs = subfigs[i].subplots(1, len(images))
        for j, img in enumerate(images):
            if j == label:
                # add a green border to the correct image (add pixels around the image)
                all_green = Image.new(
                    "RGB", (img.width + 40, img.height + 40), (0, 255, 0)
                )
                all_green.paste(img, (20, 20))
                img = all_green

            axs[j].imshow(img)
            axs[j].axis("off")

        subfigs[i].suptitle(f"{text}")

    plt.show()


def get_class_prior(dataset: PhotoBookDataset):
    class_counts = np.zeros(6)
    for data in dataset.data:
        class_counts[data["label"]] += 1

    return class_counts / class_counts.sum()


def evaluate(model, dataset: PhotoBookDataset, device="cpu"):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    tp, tn, fp, fn = 0, 0, 0, 0
    for i, data in enumerate(loader):
        text_embedding, image_embeddings, label = data
        text_embedding, image_embeddings, label = (
            text_embedding.to(device),
            image_embeddings.to(device),
            label.to(device),
        )

        outputs = model(text_embedding, image_embeddings)

        one_hot_labels = F.one_hot(label)
        tp += torch.logical_and(outputs > THRESHOLD, one_hot_labels).sum().item() + 1e-6
        tn += (
            torch.logical_and(outputs < THRESHOLD, ~one_hot_labels).sum().item() + 1e-6
        )
        fp += (
            torch.logical_and(outputs > THRESHOLD, ~one_hot_labels).sum().item() + 1e-6
        )
        fn += torch.logical_and(outputs < THRESHOLD, one_hot_labels).sum().item() + 1e-6

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * recall * precision) / (recall + precision)

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # plot_samples()
    # print(get_class_prior(PhotoBookDataset('train')))
    train_dataset = PhotoBookDataset("train")
    val_dataset = PhotoBookDataset("val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VisionLanguageTransformer(N_blocks=6, device=device)
    model.to(device)

    # load model
    # model.load_state_dict(torch.load('model.pth'))

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=LR_GAMMA, last_epoch=-1
    )
    print(
        f"Learning rate decaying exponentially with gamma {LR_GAMMA} from {LEARNING_RATE} to {scheduler.get_last_lr}"
    )

    iteration_losses = []
    train_accs, val_accs = [], []

    best_loss = np.inf

    for epoch in range(N_EPOCHS):

        model.train()
        total_loss = 0
        all_outputs = np.zeros(6)
        correct, total = 0, 0
        tp, tn, fp, fn = 0, 0, 0, 0
        for i, data in enumerate(tqdm(train_loader)):
            text_embedding, image_embeddings, label = data
            text_embedding, image_embeddings, label = (
                text_embedding.to(device),
                image_embeddings.to(device),
                label.to(device),
            )

            optimizer.zero_grad()

            outputs = model(text_embedding, image_embeddings)
            one_hot_labels = F.one_hot(label)

            labels = label.clone()
            loss = criterion(outputs, one_hot_labels.to(torch.float32))
            iteration_losses.append(loss.cpu().detach().numpy().item())

            # for each output, get the index of the highest value and increment the corresponding index in all_outputs
            for output in outputs:
                all_outputs[output.argmax().item()] += 1

            tp += (
                torch.logical_and(outputs > THRESHOLD, one_hot_labels).sum().item()
                + 1e-6
            )
            tn += (
                torch.logical_and(outputs < THRESHOLD, ~one_hot_labels).sum().item()
                + 1e-6
            )
            fp += (
                torch.logical_and(outputs > THRESHOLD, ~one_hot_labels).sum().item()
                + 1e-6
            )
            fn += (
                torch.logical_and(outputs < THRESHOLD, one_hot_labels).sum().item()
                + 1e-6
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc, val_precision, val_recall, val_f1 = evaluate(
            model, val_dataset, device
        )
        train_accs.append((tp + tn) / (tp + tn + fp + fn))
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}, Train Accuracy: {(tp + tn) / (tp + tn + fp + fn)}, Train Precision: {tp / (tp + fp)}, Train Recall: {tp / (tp + fn)}, Train F1: {(2 * tp) / (2*tp + fp + fn)}"
        )
        print(
            f"Epoch {epoch + 1}, Val Accuracy: {val_acc}, Val Precision: {val_precision}, Val Recall: {val_recall}, Val F1: {val_f1}"
        )
        print(all_outputs, all_outputs.sum())

        if best_loss > total_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), "model.pth")

        scheduler.step()

    # save model
    # torch.save(model.state_dict(), 'model.pth')

    plt.plot(iteration_losses)
    plt.show()

    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    # plot some predictions
    model.eval()
    fig = plt.figure(constrained_layout=True, figsize=(8, 12))
    subfigs = fig.subfigures(12, 1)
    indices = random.choices(range(len(val_dataset)), k=12)

    for idx in range(12):
        i = indices[idx]
        data = val_dataset.data[i]
        images = [Image.open(f"../images/{img}") for img in data["image_paths"]]
        label = data["label"]
        text = data["raw_text"]

        text_embedding, image_embeddings, label = val_dataset[i]
        text_embedding, image_embeddings = text_embedding.to(
            device
        ), image_embeddings.to(device)

        outputs = model(text_embedding.unsqueeze(0), image_embeddings.unsqueeze(0))

        prediction = outputs.argmax().item()
        confidence = outputs.max().item()

        padding = 20
        axs = subfigs[idx].subplots(1, len(images))
        for j, img in enumerate(images):
            if (
                j == label and outputs[j // 6, j % 6] > THRESHOLD
            ):  # Ground truth and model prediction match
                img = ImageOps.expand(img, border=padding, fill=(255, 255, 0))
            elif j == label:  # Image is ground truth
                img = ImageOps.expand(img, border=padding, fill=(0, 255, 0))
            elif outputs[j // 6, j % 6] > THRESHOLD:  # Image is predicted by model
                img = ImageOps.expand(img, border=padding, fill=(255, 0, 0))

            axs[j].imshow(img)
            axs[j].axis("off")

        subfigs[idx].suptitle(
            f"{text} - Prediction: {prediction} - Confidence: {confidence}"
        )
    plt.show()


#     trainset = TransformerDataset('encoded_transformed_train.json')

#     testset = TransformerDataset('encoded_transformed_test.json')

#     valset = TransformerDataset('encoded_transformed_val.json')


# model = VisionLanguageTransformer()
# #criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# load_params = {'batch_size': BATCH_SIZE,
#                'shuffle': True}

# training_loader = torch.utils.data.DataLoader(trainset, **load_params)

# # Training loop
# for epoch in range(N_EPOCHS):
#     model.train()
#     epoch_loss = 0.0
#     for i, data in enumerate(tqdm(training_loader)):
#         text_features = data['text_feature']
#         img_features = data['img_features']
#         ground_truth = torch.cat(data['ground_truth'], dim=0).to(torch.float32)
#         optimizer.zero_grad()

#         x, y, z = text_features.shape
#         outputs = model(text_features.reshape(
#             (x * y, z)).unsqueeze(1), img_features.reshape((x * y, z)).unsqueeze(1))
#         ground_truth = ground_truth.reshape(-1, 6).argmax(dim=1)
#         loss = criterion(outputs, ground_truth)

#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()

#     avg_loss = epoch_loss / len(training_loader)
#     print(f"Epoch {epoch + 1}/{N_EPOCHS}, Loss: {avg_loss:.4f}")

#     # torch.save({
#     #     'epoch': epoch + 1,
#     #     'model_state_dict': model.state_dict(),
#     #     'optimizer_state_dict': optimizer.state_dict(),
#     #     'loss': avg_loss,
#     # }, f'checkpoints/checkpoint_epoch_{epoch + 1}.pth')
