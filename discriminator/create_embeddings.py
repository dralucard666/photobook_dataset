from models.encoder import ImageEncoder, TextEncoder

from transformers import BertTokenizer
from PIL import Image
import torch

from torch.utils.data import DataLoader, TensorDataset

FILE_PATH = "..."
BATCH_SIZE = 32  # set to any value


if __name__ == "__main__":

    images: list[Image.Image] = []
    texts: list[str] = []
    images_to_text: dict[int, list[int]] = {}  # image_id -> list of text_ids

    # Load images and texts
    with open(FILE_PATH, "r") as f:
        # extract image and text data
        data_points = ...

        for data_point in data_points:
            img_path = ...
            data_point_img: Image.Image = Image.open(img_path)
            data_point_img = data_point_img.convert("RGB")
            data_point_img = data_point_img.resize((224, 224))
            data_point_texts: list[str] = ...

            images.append(data_point_img)  # add single image
            texts.extend(data_point_texts)  # add all texts

            # add image to text mapping
            img_id = len(images) - 1
            images_to_text[img_id] = list(
                range(len(texts) - len(data_point_texts), len(texts))
            )

    # convert images to tensors and permute to (batch, num_channels, img_size, img_size)
    images = [torch.tensor(image).permute(2, 0, 1) for image in images]
    images = torch.stack(images)
    image_dataset = TensorDataset(images)
    image_loader = DataLoader(image_dataset, batch_size=BATCH_SIZE)

    # tokenize texts
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    text_dataset = TensorDataset(input_ids, attention_mask)

    # create image and text encoders
    image_encoder = ImageEncoder()
    text_encoder = TextEncoder()

    # create embeddings
    image_embeddings = []
    text_embeddings = []

    for images in image_loader:
        image_embeddings.append(image_encoder(images))

    for input_ids, attention_mask in text_dataset:
        text_embeddings.append(text_encoder(input_ids, attention_mask))

    image_embeddings = torch.cat(image_embeddings)
    text_embeddings = torch.cat(text_embeddings)

    print(image_embeddings.shape)
    print(text_embeddings.shape)

    # realign images and texts and save embeddings
    image_embeddings = image_embeddings.cpu().detach().numpy()
    text_embeddings = text_embeddings.cpu().detach().numpy()

    # save embeddings
    ...
