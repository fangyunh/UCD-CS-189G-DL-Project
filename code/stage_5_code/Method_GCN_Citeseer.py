import torch.nn as nn
import torch
from code.base_class.method import method
import torch.nn.functional as F
import numpy as np
from code.stage_5_code.Evaluator import Evaluate
import matplotlib.pyplot as plt

torch.manual_seed(2)
np.random.seed(2)

class GraphConvLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        output = output + self.bias
        return output

class Method_GCN_Citeseer(method, nn.Module):

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.hidden = 1024
        self.hidden2 = 512
        self.hidden3 = 256
        self.hidden4 = 64
        self.n_class = 6

        self.conv_1 = GraphConvLayer(3703, self.hidden)
        self.conv_2 = GraphConvLayer(self.hidden, self.hidden2)
        self.conv_3 = GraphConvLayer(self.hidden2, self.hidden3)
        self.conv_4 = GraphConvLayer(self.hidden3, self.hidden4)
        self.fc_1 = nn.Linear(self.hidden4, self.n_class)
        self.dropout = 0.2

    def forward(self, x, adj):
        x = self.conv_1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv_2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv_3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv_4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc_1(x)
        return x

    def train(self, X, y, adj, train_idx, test_idx):

        max_epochs = 8
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        accuracy_evaluator = Evaluate('training evaluator', '')

        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))

        losses_per_epoch = []

        for epoch in range(max_epochs):
            self.training = True
            optimizer.zero_grad()
            y_pred = self.forward(X, adj)
            loss = loss_fn(y_pred[train_idx], y[train_idx])

            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")
            losses_per_epoch.append(loss.item())
            with torch.no_grad():
                self.training = False
                pred_test = self.forward(X, adj)
                accuracy_evaluator.data = {'true_y': y[test_idx], 'pred_y': torch.argmax(pred_test, dim=1)[test_idx]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', loss.item())


        plt.plot(losses_per_epoch)
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss Term Values")
        plt.show()

    def run(self):

        graph_data = self.data['graph']
        input_data = self.data['train_test_val']
        idx_train, idx_test = input_data['idx_train'], input_data['idx_test']
        all_inputs, all_labels = graph_data['X'], graph_data['y']
        adj = graph_data['utility']['A']

        self.train(all_inputs, all_labels, adj, idx_train, idx_test)






