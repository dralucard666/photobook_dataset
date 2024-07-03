from data_loader import TransformerDataset
from models.vision_language_transformer import VisionLanguageTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


LEARNING_RATE = 1e-4
BATCH_SIZE = 64
N_EPOCHS = 100


if __name__ == '__main__':

    trainset = TransformerDataset('encoded_transformed_train.json')

    testset = TransformerDataset('encoded_transformed_test.json')

    valset = TransformerDataset('encoded_transformed_val.json')


model = VisionLanguageTransformer()
#criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

load_params = {'batch_size': BATCH_SIZE,
               'shuffle': True}

training_loader = torch.utils.data.DataLoader(trainset, **load_params)

# Training loop
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0.0
    for i, data in enumerate(tqdm(training_loader)):
        text_features = data['text_feature']
        img_features = data['img_features']
        ground_truth = torch.cat(data['ground_truth'], dim=0).to(torch.float32)
        optimizer.zero_grad()

        x, y, z = text_features.shape
        outputs = model(text_features.reshape(
            (x * y, z)).unsqueeze(1), img_features.reshape((x * y, z)).unsqueeze(1))
        ground_truth = ground_truth.reshape(-1, 6).argmax(dim=1)
        loss = criterion(outputs, ground_truth)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(training_loader)
    print(f"Epoch {epoch + 1}/{N_EPOCHS}, Loss: {avg_loss:.4f}")

    # torch.save({
    #     'epoch': epoch + 1,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': avg_loss,
    # }, f'checkpoints/checkpoint_epoch_{epoch + 1}.pth')