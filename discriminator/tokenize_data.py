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
    index = 0

    all_objects = []

    for key in json_data:
        index = index + 1
        if index > 2:
            break
        image_path = os.path.join(image_folder, key)
        if os.path.exists(image_path):
            print('kommen wir hier hin?')
            print(json_data[key])

            object = {
                'image_path': key,
                'image': load_image(image_path),
                'messages': [message['Message_Text']
                             for message in json_data[key]]
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

        text_size = len(obj['messages'])
        print(obj['messages'])
        # tokenized = [tokenizer.tokenize(s) for s in obj['messages']]
        encoding = tokenizer(obj['messages'], padding=True,
                             truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        text_features = text_encoder(input_ids, attention_mask)

        img = torch.stack([transform(i) for i in obj['image']])
        img_features = visual_encoder(img)

        text_features = text_features.view(
            text_size, -1, text_features.size(-1))
        img_features = img_features.unsqueeze(1)

        new_obj.append({
            **obj,
            'img_features': img_features,
            'text_features': text_features,
        })

    return new_obj


def main():
    for file_name in os.listdir(json_folder):
        if '_' in file_name and file_name.endswith('.json'):
            with open(os.path.join(json_folder, file_name), 'r') as file:
                data = json.load(file)
                batches = process_data(data)

            with open(os.path.join(json_folder, 'encoded_'+file_name), 'w') as file:
                json.dump(batches, file)


if __name__ == '__main__':
    main()
