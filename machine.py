import torch
import torch.nn as nn
import torch.nn.functional as F


class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.hidden_size = 256
        self.lstm = nn.LSTM(125, 256, batch_first=True)
        self.hidden_cell = (torch.zeros(self.hidden_size), torch.zeros(self.hidden_size))
        self.linear = nn.Linear(self.hidden_size, 1)


    def forward(self, x):
        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        predictions = self.linear(lstm_out)
        return predictions[-1]