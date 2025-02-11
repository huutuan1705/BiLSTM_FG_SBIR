import torch 
import torch.nn as nn

from attention import AttentionSequence

class BiLSTM(nn.Module):
    def __init__(self, input_size, num_layers, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.hidden_size = input_size // 2 if bidirectional else input_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
            
    def forward(self, x):
        x, _ = self.bilstm(x)
        x, _ = self.bilstm(x)

        return x


# x = torch.randn(48, 25, 2048)
# model = BiLSTM(input_size=2048, num_layers=1, output_size=2048, bidirectional=True)
# output = model(x)
# output = output
# print(output.shape) # [48, 25, 2048]