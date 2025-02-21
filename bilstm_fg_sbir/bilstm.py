import torch 
import torch.nn as nn
import torch.nn.functional as F

from attention import AttentionSequence, Linear_global

class BiLSTM(nn.Module):
    def __init__(self, args, input_size=2048, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.num_bilstm_blocks = args.num_bilstm_blocks
        
        # self.hidden_size = 1024
        # self.num_layers = 1
        # self.num_bilstm_blocks = 2
        self.bilstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.attention = AttentionSequence()
        self.linear = Linear_global(feature_num=64)
        
    def forward(self, x):
        for _ in range(self.num_bilstm_blocks):
            x, _ = self.bilstm(x)
            
        x = self.attention(x)
        x = self.linear(x)      
        return x
    

# model = BiLSTM(None)
# dummy_input = torch.randn(25, 2048)
# output = model(dummy_input)
# print(output.shape)  # (25, 64)