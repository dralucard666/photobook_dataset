import os
import json
import torch
from torch import nn
import numpy as np
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms
from models.encoder import ImageEncoder, TextEncoder

# Define paths and constants
json_folder = './data/new_embedding/'
image_folder = '../images'

# Initialize encoders and tokenizer
visual_encoder = ImageEncoder()
text_encoder = TextEncoder()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def process_data(json_data):

    all_objects = []

    for key in json_data:
        image_path = os.path.join(image_folder, key)
        if os.path.exists(image_path):

            object = {
                'image_path': key,
                'image': load_image(image_path),
                'messages': [message['Message_Text']
                             for message in json_data[key]],
                'all_images': [message['All_Images']
                               for message in json_data[key]],
            }
            all_objects.append(object)

    return encode_obj(all_objects)


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image


def encode_obj(all_objects):
    new_obj = []

    for obj in all_objects:
        print(obj['image_path'])
        text_size = len(obj['messages'])
        encoding = tokenizer(obj['messages'], padding=True,
                             truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        text_features = text_encoder(input_ids, attention_mask)

        img = torch.stack([obj['image']])
        img_features = visual_encoder(img)

        new_obj.append({
            **obj,
            'image': None,
            'img_features': img_features.tolist(),
            'text_features': text_features.tolist(),
        })

    return new_obj


def main():
    for file_name in ['test', 'train', 'val']:
        file_name = 'transformed_'+file_name+'.json'
        with open(os.path.join(json_folder, file_name), 'r') as file:
            data = json.load(file)
            batches = process_data(data)
        with open(os.path.join(json_folder, 'encoded_'+file_name), 'w') as file:
            json.dump(batches, file)


if __name__ == '__main__':
    main()
