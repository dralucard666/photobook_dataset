import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import json

def extract_features(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.squeeze().cpu().numpy().tolist()

def get_image_paths(parent_dir):
    image_paths = []
    for root, _, files in os.walk(parent_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def build_image_features(parent_dir):
    resnet = models.resnet152(pretrained=True)
    resnet.fc = nn.Identity()
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = get_image_paths(parent_dir)
    features_dict = {}

    for image_path in image_paths:
        features = extract_features(image_path, resnet, transform)
        features_dict[image_path] = features

    return features_dict

if __name__ == "__main__":
    parent_dir = "transformer/data/images"
    features_dict = build_image_features(parent_dir)
    print(features_dict)

    output_file = "transformer/data/features.json"
    with open(output_file, 'w') as f:
        json.dump(features_dict, f, indent=4)

    print(f"Features saved to {output_file}")
