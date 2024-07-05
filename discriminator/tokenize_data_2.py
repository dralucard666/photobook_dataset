import json
import os

import torch
from torch.utils.data import Dataset

from tqdm import tqdm

from transformers import BertTokenizer
from torchvision import transforms
from models.encoder import ImageEncoder, TextEncoder

from PIL import Image

EMBED_BASE_PATH = 'data/new_embedding'
IMG_PATH = '../images'
IMG_SIZE = 224


class PhotoBookDataset(Dataset):

    def __init__(self, subset):
        path = os.path.join(EMBED_BASE_PATH, f'encoded_transformed_{subset}.json')
        with open(path, 'r') as f:
            self.data = json.load(f)

        with open(os.path.join(EMBED_BASE_PATH, 'image_embeddings.json'), 'r') as f:
            self.image_embeddings = json.load(f)

        print(len(self.image_embeddings))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text_embedding = torch.FloatTensor(item['text_embedding'])
        image_embeddings = [self.image_embeddings[img] for img in item['image_paths']]
        image_embeddings = torch.FloatTensor(image_embeddings)
        label = item['label']
        return text_embedding, image_embeddings, label


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image


def load_images(image_paths: list):
    images = [load_image(os.path.join(IMG_PATH, img)) for img in image_paths]
    return torch.stack(images)


def process_json(data: dict):
    """Processes a dictionary of JSON data and extracts relevant information into a list of dictionaries.

    Args:
        data (dict): A dictionary where keys are image identifiers and values are lists of dictionaries. 
                     Each inner dictionary contains 'Message_Text' and 'All_Images' keys.

    Returns:
        list: A list of dictionaries, each containing the keys:
              - 'correct_image': The key from the original dictionary.
              - 'text': The 'Message_Text' from the inner dictionary.
              - 'all_images': The 'All_Images' from the inner dictionary.
    """
    objects = []
    all_images = []

    print("Processing raw json data...")
    for key, value in tqdm(data.items()):
        for item in value:
            object = {
                'correct_image': key,
                'text': item['Message_Text'],
                'all_images': item['All_Images']
            }
            objects.append(object)
            all_images.extend(item['All_Images'])

    return objects, list(set(all_images))


def create_embeddings(objects: list):
    """Generates embeddings for both text and images from a list of objects containing text and image data.

    Args:
        objects (list): A list of dictionaries. Each dictionary is one sample containing 6 images and one
                        corresponding text. The dictionary has the following keys:
                        - 'text': The text data to be embedded.
                        - 'all_images': A list of image file names.
                        - 'correct_image': The correct image file name.

    Returns:
        list: A list of dictionaries, each containing:
              - 'raw_text': The original text.
              - 'text_embedding': The embedding of the text.
              - 'image_paths': The list of image file names.
              - 'image_embeddings': The list of image embeddings.
              - 'label': The index of the correct image in the 'all_images' list.

    Note:
        In this setting every enty in the returned list corresponds to one sample for the task of predicting
        the correct image given the text embeddings and the embeddings of the 6 images.
    """
    text_encoder = TextEncoder()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    new_objects = []
    for obj in tqdm(objects):
        text = obj['text']
        all_images = obj['all_images']
        correct_image = obj['correct_image']

        tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        text_embed = text_encoder(tokens['input_ids'], tokens['attention_mask'])
        
        images = [load_image(os.path.join(IMG_PATH, img)) for img in all_images]
        images = torch.stack(images)

        new_obj = {
            'raw_text': text,
            'text_embedding': text_embed.detach().numpy().tolist(),
            'image_paths': all_images,
            'label': all_images.index(correct_image)
        }

        new_objects.append(new_obj)

    return new_objects


def create_image_embeddings(images: list, batch_size: int = 64):
    visual_encoder = ImageEncoder()

    embeddings = []
    for i in tqdm(range(0, len(images), batch_size)):
        img_names = images[i:i + batch_size]
        img_tensors = load_images(img_names)

        img_embeds = visual_encoder(img_tensors)

        embeddings.extend(img_embeds.detach().numpy().tolist())

    # create a dictionary with image file names as keys and embeddings as values
    img_embed_dict = dict(zip(images, embeddings))
    return img_embed_dict


def main():
    all_images = []
    sets = ['train', 'val', 'test']

    for subset in sets:
        path = os.path.join(EMBED_BASE_PATH, f'transformed_{subset}.json')

        with open(path, 'r') as f:
            data = json.load(f)

        objects, imgs = process_json(data)
        embeddings = create_embeddings(objects)
        all_images.extend(imgs)

        save_path = os.path.join(EMBED_BASE_PATH, f'encoded_transformed_{subset}.json')

        with open(save_path, 'w') as f:
            json.dump(embeddings, f)

    all_images = list(set(all_images))
    img_embeds = create_image_embeddings(all_images)

    with open(os.path.join(EMBED_BASE_PATH, 'image_embeddings.json'), 'w') as f:
        json.dump(img_embeds, f)


if __name__ == '__main__':
    main()
    dataset = PhotoBookDataset('train')

