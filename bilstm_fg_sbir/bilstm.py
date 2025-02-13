import torch 
import torch.nn as nn

from attention import AttentionSequence

class BiLSTM(nn.Module):
    def __init__(self, input_size=2048, num_layers=1, hidden_size1 = 512, hidden_size2 = 32, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers
        self.bilstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.input_size // 2, num_layers=self.num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.bilstm2 = nn.LSTM(input_size=self.input_size, hidden_size=self.input_size // 2, num_layers=self.num_layers,
                            batch_first=True, bidirectional=bidirectional)
    def forward(self, x):
        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)
        
        return x


# x = torch.randn(48, 25, 2048)
# model = BiLSTM(input_size=2048, num_layers=2, bidirectional=True)
# output = model(x)
# output = output
# print(output.shape) # [48, 25, 64]