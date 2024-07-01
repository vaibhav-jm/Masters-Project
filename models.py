import torch 
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        # calls parent class constructor
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        # hidden_size * 2 because we are using a bidirectional RNN (double the length of the hidden state)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.droput = nn.Dropout(0.3)
        # self.bn = nn.BatchNorm1d(hidden_size*2)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # out = out.permute(0, 2, 1)
        # out = self.bn(out)
        # sequence-to-sequence classification problem where you want to classify each time step in the sequence independently.
        # Pass each time step's output through a fully connected layer
        out = self.fc(out)
        out = self.droput(out)
        return out

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        # calls parent class constructor
        super(BiRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # hidden_size * 2 because we are using a bidirectional RNN (double the length of the hidden state)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.droput = nn.Dropout(0.3)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        # Pass each time step's output through a fully connected layer
        out = self.fc(out)
        out = self.droput(out)
        # out = torch.sigmoid(out)
        return out

def load_model(model_path):
    model = torch.load(f'{model_path}.pth')
    return model