import os
import numpy as np
root_path = '../../data/stage_4_data/text_classification'

reviews = {}
labels = {}
lab = {"pos": 1, "neg": 0}
for set in ["train", "test"]:
    text_set = []
    lab_set = []
    for l in ["pos", "neg"]:
        path = root_path + f"/{set}/{l}"
        for f in os.listdir(path):
            if f.endswith(".txt"):
                with open(path + "/" + f, 'r', encoding='utf-8') as file:
                    rev = file.read()
                    rev_lab = lab[l]
                    text_set.append(rev)
                    lab_set.append(rev_lab)
        reviews[set] = np.array(text_set)
        labels[set] = np.array(lab_set)
print(reviews["train"][1])
print(labels["train"][1])