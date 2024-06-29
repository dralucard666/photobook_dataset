import json
import torch


def load_objects_from_file(file_path):
    objects = []

    with open(file_path, 'r') as f:
        data = json.load(f)

        for obj in data:
            # Convert lists back to tensors
            if 'img_features' in obj:
                obj['img_features'] = torch.tensor(obj['img_features'])
            if 'text_features' in obj:
                obj['text_features'] = torch.tensor(obj['text_features'])

            objects.append(obj)

    return objects


def main():
    file_path = './data/new_embedding/encoded_transformed_test.json'
    loaded_objects = load_objects_from_file(file_path)

    # Example usage: print the first object's img_features tensor
    if loaded_objects:
        print(
            f"First object's img_features tensor: {loaded_objects[0]['img_features']}")
        print(
            f"First object's text_features tensor: {loaded_objects[0]['text_features']}")
    else:
        print("No objects found in the file.")


if __name__ == "__main__":
    main()
