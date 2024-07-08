import os
import json
import string


def extract_history_features(data):
    features = {}
    punctuation_marks = set(string.punctuation)
    
    for round in data["rounds"]:
        history = ""
        seen = set()
        for message in round["messages"]:
            msg = message["message"].rstrip()

            if "feedback" in msg:
                continue

            if "<selection>" in msg:
                if "<com>" in msg:
                    sel_image = msg[18:]
                elif "<dif>" in msg:
                    sel_image = msg[18:]

                if sel_image in seen:
                    continue

                seen.add(sel_image)

                if history == "":
                    continue

                feature = {
                    "Game_ID": data["game_id"],
                    "Message_Speaker": "AB",
                    "Message_Text": history.rstrip(),
                    "All_Images": round["images"][message["speaker"]]
                }

                if sel_image in features.keys():
                    l = features[sel_image]
                    l.append(feature)
                    features[sel_image] = l
                else:
                    l = [feature]
                    features[sel_image] = l

                history = ""

                continue

            if msg and not msg[-1] in punctuation_marks:
                msg += "."
            history += msg + " "
    return features


def iterate_raw_data(path, train_test_split=.8):
    files = os.listdir(path)

    all_features = {}
    for filename in files[:int(len(files)*train_test_split)]:
        file_path = os.path.join(path, filename)
        if not os.path.isfile(file_path):
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

            features = extract_history_features(data)

            for key, value in features.items():
                if key in all_features.keys():
                    l = [*all_features[key], *value]
                    all_features[key] = l
                else:
                    all_features[key] = value

    all_features_test = {}
    for filename in files[int(len(files)*train_test_split):]:
        file_path = os.path.join(path, filename)
        if not os.path.isfile(file_path):
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

            features = extract_history_features(data)

            for key, value in features.items():
                if key in all_features_test.keys():
                    l = [*all_features_test[key], *value]
                    all_features_test[key] = l
                else:
                    all_features_test[key] = value

    return all_features, all_features_test


if __name__ == '__main__':
    dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(dir, 'train.json')
    save_path_test = os.path.join(dir, 'test.json')
    dir = os.path.join(dir, 'logs')

    features, features_test = iterate_raw_data(dir)
    
    with open(save_path, 'w') as f:
        json.dump(features, f, indent=4)
    
    with open(save_path_test, 'w') as f:
        json.dump(features_test, f, indent=4)
