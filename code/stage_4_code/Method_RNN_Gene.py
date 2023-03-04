from code.base_class.method import method
import torch
import torch.nn as nn
import numpy as np
import torchtext
from collections import defaultdict
import matplotlib.pyplot as plt

torch.manual_seed(2)
np.random.seed(2)

class Method_RNN_Generalization(method, nn.Module):

    def __init__(self, mName, mDescription, vocab_w_i, vocab_i_w):
        nn.Module.__init__(self)

        self.hidden_dim = 200
        self.embedding_dim = 250
        self.num_layers = 2
        self.dropout = nn.Dropout(0.5)

        vocab_size = len(vocab_w_i)

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        #self.rnn = nn.RNN(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        #self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.dropout(self.embedding(x))
        output, hidden_out = self.rnn(embedded, hidden)
        output = self.fc(output)
        output = output[:,-1,:]
        return output, hidden_out

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_dim),
                torch.zeros(self.num_layers, sequence_length, self.hidden_dim))

    def train(self, X, y, jokes, jokes_len, vocab_w_i, vocab_i_w):

        words_not_gloved = 0
        glove = torchtext.vocab.GloVe(name="6B", dim=100)
        for word in list(vocab_w_i.keys()):
            try:
                self.embedding.weight.data[vocab_w_i[word]] = glove.vectors[glove.stoi[word]]
            except:
                self.embedding.weight.data[vocab_w_i[word]] = torch.zeros(self.embedding_dim)
                words_not_gloved += 1
        self.embedding.weight.data[0] = torch.zeros(self.embedding_dim)
        self.embedding.weight.data[1] = torch.zeros(self.embedding_dim)

        X = torch.LongTensor(X)
        y = torch.LongTensor(y)
        batch_size = 300
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        loss_function = torch.nn.CrossEntropyLoss()

        per_sentences = []
        prev = 0
        prev_res = 0
        epochs = 10
        for sent in range(10, len(jokes_len), 10):
            res = sum(jokes_len[prev:sent]) - 3 * 10
            prev += 10
            range_tuple = (prev_res, prev_res + res)
            prev_res = res + prev_res
            per_sentences.append(range_tuple)

        print(jokes[9])
        loss_val = []
        epoch_set = []

        for epoch in range(epochs):
            hidden_state, cell_state = self.init_state(sequence_length=46)

            for i in range(0, X.size()[0], batch_size):
                batch_x = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                optimizer.zero_grad()
                y_pred, h = self.forward(batch_x, None)
                loss = loss_function(y_pred, batch_y)

                hidden_state = hidden_state.detach()
                cell_state = cell_state.detach()

                loss.backward()
                optimizer.step()

                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
            loss_val.append(loss.item())
            epoch_set.append(epoch)
        plt.plot(epoch_set, loss_val)
        plt.ylabel('Loss Term Values')
        plt.xlabel('Training Epoch')
        plt.show()

        joke_tracker = defaultdict(int)
        total_jokes_corr = 0
        non_unique = 0

        for i in range(len(jokes)):
            joke_tracker[str(jokes[i][:3])] += 1

        for i in range(len(jokes)):
            if joke_tracker[str(jokes[i][:3])] > 1:
                non_unique += 1
                continue
            pred = self.generate_text(jokes[i][:3], vocab_w_i, vocab_i_w)
            real = jokes[i]
            if pred == real:
                total_jokes_corr += 1

    def generate_text(self, input_text, vocab_w_i, vocab_i_w):
        self.training = False
        text = input_text
        length = 20

        words = text + []

        for i in range(0, length-len(text)):
            tokenized = torch.tensor([vocab_w_i[word] for word in text]).unsqueeze(0)
            prediction, h = self.forward(tokenized, None)
            prediction = prediction.squeeze(0)

            next_word = torch.argmax(prediction).item()
            words.append(vocab_i_w[next_word])
            text = text[1:] + [vocab_i_w[next_word]]
            print("text: ", text)
        print("Predict: ", words)
        return words

    def run(self):
        self.train(self.data['X'], self.data['y'], self.data['jokes'], self.data['jokes_len'],
                   self.data['w_i'], self.data['i_w'])
        self.generate_text(['my', 'friend', 'told'], self.data['w_i'], self.data['i_w'])


