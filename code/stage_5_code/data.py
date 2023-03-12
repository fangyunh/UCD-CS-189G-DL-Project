'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def sampling(self, features, classes, label, sample_num):
        idx = np.array(range(features.size()[0]))
        idx_train = []
        idx_test = []

        for i in range(classes):
            idx_train += list(np.random.choice(idx[np.where(label[idx] == i)[0]], 20, replace=False))

        test_sample = np.array(list(set(list(idx)) - set(idx_train)))

        for i in range(classes):
            if len(test_sample[np.where(label[test_sample] == i)[0]]) <= sample_num:
                idx_test += list(test_sample[np.where(label[test_sample] == i)[0]])
            else:
                chosen_indices = np.random.choice(test_sample[np.where(label[test_sample] == i)[0]], sample_num,
                                                  replace=False)
                idx_test += list(chosen_indices)

        return idx_train, idx_test

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))
        numbers_of_data = 1433
        data_path = self.dataset_source_folder_path + self.dataset_source_file_name
        print(data_path)

        node_names = ["Node"] + [f"word_{i}" for i in range(numbers_of_data)] + ["category"]
        node_data = pd.read_csv("{}/node".format(data_path), sep="\t", names=node_names)
        column_names = ["target", "source"]
        edge_data = pd.read_csv("{}/link".format(data_path), sep="\t", names=column_names)
        class_dict = {c: i for i, c in enumerate(node_data.category.unique())}
        node_data['category'] = [class_dict[i] for i in node_data['category']]
        node_data = node_data.to_numpy()
        edge_data = edge_data.to_numpy()

        # load node data from file
        idx_features_labels = np.genfromtxt("{}/node".format(data_path), dtype=np.dtype(str))
        # print(idx_features_labels)
        features = sp.csr_matrix(node_data[:, 1:-1], dtype=np.float32)
        # onehot_labels = self.encode_onehot(idx_features_label[:, -1])
        onehot_labels = np.array(node_data[:, -1], dtype=np.int32)
        # print(onehot_labels)

        # load link data from file and build graph
        idx = np.array(node_data[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.array(edge_data, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels))
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # the following train, test, val index are just examples, sample the train, test according to project requirements
        max_instances = 150

        if self.dataset_source_file_name == 'cora':
            classes = 7
            train_idx, test_idx = self.sampling(features, classes, onehot_labels, max_instances)
        elif self.dataset_source_file_name == 'citeseer':
            classes = 6
            train_idx, test_idx = self.sampling(features, classes, onehot_labels, max_instances)
        elif self.dataset_source_file_name == 'pubmed':
            classes = 3
            train_idx, test_idx = self.sampling(features, classes, onehot_labels, max_instances)
        else:
            print("Not a valid dataset.")

        idx_train = torch.LongTensor(np.array(train_idx))
        idx_test = torch.LongTensor(test_idx)
        print(f"what is {idx_train}")
        # idx_val = torch.Tensor(class_dict)
        # get the training nodes/testing nodes
        # train_x = features[idx_train]
        # val_x = features[idx_val]
        # test_x = features[idx_test]
        # print(train_x, val_x, test_x)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels,
                 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}, class_dict

