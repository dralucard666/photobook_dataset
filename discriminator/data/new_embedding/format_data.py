import os
import json


def transform_data(input_data):
    transformed_data = {}
    for image_key, game_entries in input_data.items():
        filtered_entries = []
        for entry_id, entry_list in game_entries.items():
            for entry in entry_list:
                if entry.get("Reason") == "<com>":
                    speaker = entry["Message_Speaker"]
                    filtered_entry = {
                        "Game_ID": entry["Game_ID"],
                        "Message_Speaker": entry["Message_Speaker"],
                        "Message_Text": entry["Message_Text"],
                        "All_Images": speaker == "B"
                    }
                    if speaker == "A":
                        filtered_entry['All_Images'] = entry["Round_Images_A"]
                    else:
                        filtered_entry['All_Images'] = entry["Round_Images_B"]

                    filtered_entries.append(filtered_entry)
        if filtered_entries:
            transformed_data[image_key] = filtered_entries
    return transformed_data


def process_json_files(folder_path):
    for filename in ['test.json', 'train.json', 'val.json']:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            transformed_data = transform_data(data)
            output_file_path = os.path.join(
                folder_path, f"transformed_{filename}")
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                json.dump(transformed_data, output_file, indent=4)


if __name__ == "__main__":
    folder_path = os.path.dirname(os.path.realpath(__file__))
    process_json_files(folder_path)
