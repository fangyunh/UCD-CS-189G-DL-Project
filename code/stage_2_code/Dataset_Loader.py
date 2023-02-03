'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')

        f = open(self.dataset_source_folder_path + "train.csv", 'r')
        data_train = np.loadtxt(f, delimiter=",")
        X_train = data_train[:, 1:]
        y_train = data_train[:, 0]
        f.close()

        f = open(self.dataset_source_folder_path + "test.csv", 'r')
        data_test = np.loadtxt(f, delimiter=",")
        X_test = data_test[:, 1:]
        y_test = data_test[:, 0]
        f.close()
        return {'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test}