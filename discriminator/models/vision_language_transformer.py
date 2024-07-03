import torch
from torch import nn
import numpy as np

from transformers import BertTokenizer
from torchvision import transforms


class VisionLanguageTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768, num_heads=8)
        self.dense1 = nn.Linear(768, 256)
        self.dense2 = nn.Linear(256, 1)

    def forward(self, text_features, img_features):
        # Apply cross attention

        text_features = text_features.permute(1, 0, 2)
        img_features = img_features.permute(1, 0, 2)
        cross_attended_features, _ = self.cross_attention(
            text_features, img_features, img_features)

        # Pooling
        pooled_features = cross_attended_features.mean(dim=0)

        # Feedforward Layers
        x = self.dense1(pooled_features)
        x = nn.ReLU()(x)
        x = self.dense2(x)
        x = x.squeeze(-1)

        #x = nn.Sigmoid()(x)
        x = x.reshape(-1, 6)
        x = nn.Softmax(dim=1)(x)
        return x


if __name__ == '__main__':
    batch_size = 6
    sentence = "This is a test sentence"
    img = torch.randn(batch_size, 3, 244, 244)

    visual_encoder = ImageEncoder()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_encoder = TextEncoder()

    transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    img = torch.stack([transform(i) for i in img])

    encoding = tokenizer([sentence] * batch_size, padding=True,
                         truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    text_features = text_encoder(input_ids, attention_mask)

    img_features = visual_encoder(img)

    text_features = text_features
    img_features = img_features

    batch = (text_features, img_features)

    model = VisionLanguageTransformer()

    outputs = model(text_features, img_features)
    print(outputs)
