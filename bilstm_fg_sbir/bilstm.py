import torch 
import torch.nn as nn
import torch.nn.functional as F

from attention import AttentionSequence, Linear_global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    def __init__(self, args, input_size=2048, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.num_bilstm_blocks = args.num_bilstm_blocks
        # self.num_layers = 2
        # self.hidden_size = 1024
        # self.num_bilstm_blocks = 2
        self.bilstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        for _ in range(self.num_bilstm_blocks):
            x, _ = self.bilstm(x) # (N, 25, 2048)
        
        x = AttentionSequence().to(device)(x)
        x = Linear_global(feature_num=64).to(device)(x)
        
        return x # (N, 25, 64)  

# x = torch.randn(48, 25, 2048)
# model = BiLSTM(None, input_size=2048)
# x = model(x)
# print(x.shape) # (N,25, 64)