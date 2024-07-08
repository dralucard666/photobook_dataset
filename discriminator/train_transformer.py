from data_loader import TransformerDataset
from models.vision_language_transformer import VisionLanguageTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np

from tokenize_data_2 import PhotoBookDataset
from torch.utils.data import DataLoader

from PIL import Image
import matplotlib.pyplot as plt


LEARNING_RATE = 1e-4
BATCH_SIZE = 256
N_EPOCHS = 100


def plot_samples(num_samples=5):
    dataset = PhotoBookDataset('train')
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    subfigs = fig.subfigures(num_samples, 1)

    indices = random.choices(range(len(dataset)), k=num_samples)

    for i in range(num_samples):
        data = dataset.data[indices[i]]
        images = [Image.open(f'../images/{img}') for img in data['image_paths']]
        label = data['label']
        text = data['raw_text']
        # create a subplot showing all images next to each other
        axs = subfigs[i].subplots(1, len(images))
        for j, img in enumerate(images):
            if j == label:
                # add a green border to the correct image (add pixels around the image)
                all_green = Image.new('RGB', (img.width + 40, img.height + 40), (0, 255, 0))
                all_green.paste(img, (20, 20))
                img = all_green

            axs[j].imshow(img)
            axs[j].axis('off')

        subfigs[i].suptitle(f"{text}")

    plt.show()


def get_class_prior(dataset: PhotoBookDataset):
    class_counts = np.zeros(6)
    for data in dataset.data:
        class_counts[data['label']] += 1

    return class_counts / class_counts.sum()


def evaluate(model, dataset: PhotoBookDataset):
    model.eval()
    correct = 0
    total = 0
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    for i, data in enumerate(loader):
        text_embedding, image_embeddings, label = data

        outputs = model(text_embedding, image_embeddings)

        correct += (outputs.argmax(dim=1) == label).sum().item()
        total += len(label)

    return correct / total



if __name__ == '__main__':
    #plot_samples()
    #print(get_class_prior(PhotoBookDataset('train')))
    train_dataset = PhotoBookDataset('train')
    train_dataset.shuffle_images = True
    val_dataset = PhotoBookDataset('val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VisionLanguageTransformer()

    # load model
    #model.load_state_dict(torch.load('model.pth'))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    iteration_losses = []
    train_accs, val_accs = [], []

    for epoch in range(100):
        if epoch == 10:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        model.train()
        total_loss = 0
        all_outputs = np.zeros(6)
        correct, total = 0, 0
        for i, data in enumerate(tqdm(train_loader)):
            text_embedding, image_embeddings, label = data

            optimizer.zero_grad()

            outputs = model(text_embedding, image_embeddings)

            labels = torch.tensor(label)
            loss = criterion(outputs, labels)
            iteration_losses.append(loss.detach().numpy().item())

            # for each output, get the index of the highest value and increment the corresponding index in all_outputs
            for output in outputs:
                all_outputs[output.argmax().item()] += 1

            correct += (outputs.argmax(dim=1) == label).sum().item()
            total += len(label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, val_dataset)
        train_accs.append(correct / total)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}, Train Accuracy: {correct / total}, Val Accuracy: {val_acc}")
        print(all_outputs, all_outputs.sum())

    # save model
    # torch.save(model.state_dict(), 'model.pth')

    plt.plot(iteration_losses)
    plt.show()

    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    # plot some predictions
    model.eval()
    fig = plt.figure(constrained_layout=True, figsize=(8, 12))
    subfigs = fig.subfigures(12, 1)
    indices = random.choices(range(len(val_dataset)), k=12)

    for idx in range(12):
        i = indices[idx]
        data = val_dataset.data[i]
        images = [Image.open(f'../images/{img}') for img in data['image_paths']]
        label = data['label']
        text = data['raw_text']

        axs = subfigs[idx].subplots(1, len(images))
        for j, img in enumerate(images):
            if j == label:
                all_green = Image.new('RGB', (img.width + 40, img.height + 40), (0, 255, 0))
                all_green.paste(img, (20, 20))
                img = all_green

            axs[j].imshow(img)
            axs[j].axis('off')

        text_embedding, image_embeddings, label = val_dataset[i]

        outputs = model(text_embedding.unsqueeze(0), image_embeddings.unsqueeze(0))

        subfigs[idx].suptitle(f"{text} - Prediction: {outputs.argmax().item()} - Confidence: {outputs.max().item()}")
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