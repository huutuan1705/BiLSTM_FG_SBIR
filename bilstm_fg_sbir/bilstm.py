import torch 
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, args, input_size=2048, hidden_size1 = 512, hidden_size2 = 32, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.args = args
        self.input_size = input_size
        # self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        # self.num_layers = 2
        # self.num_bilstm_blocks = args.num_bilstm_blocks
        self.bilstm1 = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size1, num_layers=self.num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.bilstm2 = nn.LSTM(input_size=hidden_size1*2, hidden_size=hidden_size2, num_layers=self.num_layers,
                               batch_first=True, bidirectional=bidirectional)
    def forward(self, x):
        # for _ in range(self.num_bilstm_blocks):
        #     x, _ = self.bilstm(x)
        x, _ = self.bilstm1(x) # (N, 25, 1024)    
        x, _ = self.bilstm2(x) # (N, 25, 64)    
        x = x[:, -1, :]       
        return x # (N, 1, 64)  

# x = torch.randn(48, 25, 2048)
# model = BiLSTM(None, input_size=2048)
# x = model(x)
# print(x.shape) # (N, 64)