import torch 
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, args, input_size=2048, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.num_bilstm_blocks = args.num_bilstm_blocks
        self.bilstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=bidirectional)
    def forward(self, x):
        for _ in range(self.num_bilstm_blocks):
            x, _ = self.bilstm(x)
            
        x = x[:, -1, :]       
        return x