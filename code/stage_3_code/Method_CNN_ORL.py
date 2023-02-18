from code.base_class.method import method
from code.stage_3_code.Evaluator import Evaluate
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Method_CNN_ORL(method, nn.Module):
    data = None
    learning_rate = 1e-03
    epochs = 100

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv_1 = nn.Conv2d(3, 64, 3)
        self.pool_1 = nn.MaxPool2d(2, 2)
        #self.pool_1 = nn.AvgPool2d(2, 2)
        self.conv_2 = nn.Conv2d(64, 128, 3)
        self.pool_2 = nn.MaxPool2d(2, 2)
        #self.pool_2 = nn.AvgPool2d(2, 2)

        self.fc_1 = nn.Linear(69888, 1024)
        self.fc_2 = nn.Linear(1024, 128)
        self.fc_3 = nn.Linear(128, 40)

    def forward(self, X):

        x = F.relu(self.conv_1(X))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)

        return x

    def train(self, X, y):
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        evaluator = Evaluate('evaluator', '')
        batch_size = 20

        X = torch.FloatTensor(np.array(X)).permute(0, 3, 1, 2)
        y = torch.LongTensor(np.array(y))
        permutation = torch.randperm(X.size()[0])
        loss_val = []
        epoch_set = []

        for epoch in range(1, self.epochs + 1):
            for i in range(0, X.size()[0], batch_size):
                optimizer.zero_grad()

                idx = permutation[i:i + batch_size]
                xbatch, ybatch = X[idx], y[idx]
                xbatch, ybatch = xbatch.float().requires_grad_(), ybatch

                y_pred = self.forward(xbatch)
                train_loss = loss_function(y_pred, ybatch)
                train_loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                with torch.no_grad():
                    X_test = torch.FloatTensor(np.array(self.data['test']['X'])).permute(0, 3, 1, 2)
                    pred_test = self.forward(X_test)
                    evaluator.data = {'true_y' : self.data['test']['y'], 'pred_y' : torch.argmax(pred_test, dim=1)}
                    print('Epoch:', epoch, 'Score:', evaluator.evaluate(), 'Loss:', train_loss.item())

            loss_val.append(train_loss.item())
            epoch_set.append(epoch)

        plt.plot(epoch_set, loss_val)
        plt.ylabel('Loss Term Values')
        plt.xlabel('Training Epoch')
        plt.show()

    def test(self, X):
        X = torch.from_numpy(X)
        X = X.permute(0, 3, 1, 2)
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return torch.argmax(y_pred, axis=1)

    def run(self):
        #self.train(self.data['train']['X'], self.data['train']['y'])
        self.train(self.data['test']['X'], self.data['test']['y'])
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
