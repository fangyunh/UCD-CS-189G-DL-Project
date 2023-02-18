from code.base_class.dataset import dataset
import pickle
import numpy as np
import torch

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):

        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)

        train_img = []
        train_lab = []
        test_img = []
        test_lab = []

        for image_lab in data["train"]:
            image = image_lab["image"]
            label = image_lab["label"]
            train_img.append(image)
            train_lab.append(label)

        for image_lab in data["test"]:
            image = image_lab["image"]
            label = image_lab["label"]
            test_img.append(image)
            test_lab.append(label)

        train_img = torch.Tensor(np.array(train_img))
        train_lab = np.array(train_lab)-1
        test_img = torch.Tensor(np.array(test_img))
        test_lab = np.array(test_lab)-1

        train_img /= 255
        test_img /= 255

        return {"X_train": train_img, "y_train": train_lab,
                "X_test": test_img, "y_test": test_lab}