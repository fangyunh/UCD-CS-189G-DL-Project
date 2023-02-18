import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
class MNISTDataLoad(Dataset):
    dataset_source_folder_path = None
    dataset_source_file_name = None
    def __init__(self, is_train=True, transform=None):
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        self.mnist = pickle.load(f)
        self.is_train = is_train
        f.close()

    def __len__(self):
        if self.is_train:
            return int(len(self.mnist) * 6 / 7)
        else:
            return int(len(self.mnist) * 1 / 7)

    def __getitem__(self, idx):
        if self.is_train:
            idx = idx
        else:
            idx = int(len(self.mnist) * 6 / 7) + idx
        image, label = self.mnist[idx]
        image = image.unsqueeze(0)
        return image, label

    def load(self):
        trainset = MNISTDataLoad(is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True, num_workers=2)
        testset = MNISTDataLoad(is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=2)
        return trainloader, testloader
