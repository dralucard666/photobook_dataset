import json
import torch
from torch.utils.data import Dataset, DataLoader
from models.vision_language_transformer import VisionLanguageTransformer

# Custom Dataset class


class TransformerDataset(Dataset):
    def __init__(self, file_path):
        with open('./data/new_embedding/' + file_path, 'r') as file:
            self.data = json.load(file)

        self.unrolled_data = []

        for item in self.data:
            # Assuming 'image_path' is the key in your JSON data
            key = item['image_path']
            item['img_features'] = torch.FloatTensor(item['img_features'][0])
            text_features = item['text_features']
            all_images_keys = item['all_images']
            for idx in range(len(text_features)):
                text_feature = text_features[idx]
                text_feature = torch.FloatTensor(text_feature).repeat(6, 1)
                self.unrolled_data.append({
                    'key': key,
                    'text_feature': text_feature,
                    'image_keys': all_images_keys[idx],
                    'index': idx
                })

    def __len__(self):
        return len(self.unrolled_data)

    def __getitem__(self, idx):
        unrolled_item = self.unrolled_data[idx]
        key = unrolled_item['key']
        index = unrolled_item['index']
        text_feature = unrolled_item['text_feature']
        image_keys = unrolled_item['image_keys']

        # Retrieve image features using the image keys
        img_features = torch.stack([self.data[self._find_index(
            img_key)]['img_features'] for img_key in image_keys])
        ground_truth = [1 if img_key == key else 0 for img_key in image_keys]

        return {
            'key': key,
            'text_feature': text_feature,
            'img_features': img_features,
            'image_keys': image_keys,
            'ground_truth': ground_truth
        }

    def _find_index(self, img_key):
        for i, item in enumerate(self.data):
            if item['image_path'] == img_key:  # Assuming 'image_path' is the key in your JSON data
                return i
        raise ValueError(f"Image key {img_key} not found in data")

# Main function


def main():
    # Create an instance of the dataset and a DataLoader
    file_path = 'encoded_transformed_test.json'  # Replace with your file path
    game_dataset = TransformerDataset(file_path)
    data_loader = DataLoader(game_dataset, batch_size=1, shuffle=True)

    # Iterate over the data using the DataLoader and print only the first two batches
    for i, batch in enumerate(data_loader):
        if i >= 1:
            break
        key = batch['key']
        text_feature = batch['text_feature']
        img_features = batch['img_features']
        img_features = img_features.squeeze(0)
        image_keys = batch['image_keys']
        ground_truth = batch['ground_truth']
        ground_truth = torch.cat(ground_truth, dim=0)

        model = VisionLanguageTransformer()

        outputs = model(text_feature[0], img_features)
        print(outputs)
        print(ground_truth)


if __name__ == "__main__":
    main()
