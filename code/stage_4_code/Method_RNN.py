from code.base_class.method import method
import torch
import torch.nn as nn
import numpy as np
import torchtext
import matplotlib.pyplot as plt


np.random.seed(2)
torch.manual_seed(2)
class Method_RNN_Classification(method, nn.Module):
    data = None
    learning_rate = 1e-03
    dictionary = None

    def __init__(self, mName, mDescription, vocabulary_size):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.embedding_dim = 250
        self.hidden_dim = 200
        self.output_dim = 1
        self.sequence_length = 200
        self.vocabulary_size = vocabulary_size
        self.num_layers = 3

        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dim)
        #self.rnn = nn.RNN(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        #self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(0.5)

        self.word_dict = None

    def forward(self, text, hid):

        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded, hid)
        batch_size = output.shape[0]
        output = self.dropout(output.contiguous().view(-1, self.hidden_dim))
        output = self.fc(output)
        output = output.view(batch_size, -1)
        return output[:, -1], hidden

    def train(self, X, y, vocabulary):
        glove = torchtext.vocab.GloVe(name="6B", dim=100)
        for word in list(vocabulary.keys()):
            try:
                self.embedding.weight.data[vocabulary[word]] = glove.vectors[glove.stoi[word]]
            except:
                self.embedding.weight.data[vocabulary[word]] = torch.zeros(self.embedding_dim)

        self.embedding.weight.data[0] = torch.zeros(self.embedding_dim)
        self.embedding.weight.data[1] = torch.zeros(self.embedding_dim)

        X = torch.LongTensor(X)
        y = torch.FloatTensor(y)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.CrossEntropyLoss()
        batch_size = 128
        loss_val = []
        epoch_set = []

        epochs = 3
        for epoch in range(epochs):
            self.training = True
            for i in range(0, X.size()[0], batch_size):
                optimizer.zero_grad()
                xbatch, ybatch = X[i:i + batch_size, :], y[i:i+batch_size]
                output, hidden = self.forward(xbatch, None)
                output = output.squeeze(-1)
                loss = loss_function(output, ybatch)
                loss.backward()
                optimizer.step()

            if True:
                with torch.no_grad():
                    self.training = False
                    total_corr = 0
                    for j in range(0, X.size()[0], batch_size):

                        test_batch_x, test_batch_y = X[j:j+batch_size,:], y[j:j+batch_size]

                        pred_test, h_out = self.forward(test_batch_x, None)
                        pred_test = pred_test.squeeze(-1)
                        rounded_preds = torch.round(torch.sigmoid(pred_test))
                        correct = np.array(rounded_preds) == np.array(test_batch_y)
                        acc = sum(correct) / len(correct)
                        total_corr += sum(correct)
                        print('Epoch:', epoch, 'Batch:', j, '  Accuracy:', acc, 'Loss:', loss.item())
            loss_val.append(loss.item())
            epoch_set.append(epoch)

        plt.plot(epoch_set, loss_val)
        plt.ylabel('Loss Term Values')
        plt.xlabel('Training Epoch')
        plt.show()

    def test(self, X):

        if self.training == False:
            self.training = True

        X = torch.LongTensor(X)
        batch_size = 128
        predictions = np.array([])
        with torch.no_grad():
            for i in range(0, X.size()[0], batch_size):
                xbatch = X[i:i + batch_size, :]
                pred_test, h_out = self.forward(xbatch, None)
                pred_test = pred_test.squeeze(-1)
                rounded_preds = np.array(torch.round(torch.sigmoid(pred_test)))
                predictions = np.append(predictions, rounded_preds)

        return predictions

    def run(self):
        #self.train(self.data['train']['X'], self.data['train']['y'], self.dictionary)
        self.train(self.data['test']['X'], self.data['test']['y'], self.dictionary)

        predict = self.test(self.data['test']['X'])
        return {'pred_y' : predict, 'true_y' : self.data['test']['y']}

