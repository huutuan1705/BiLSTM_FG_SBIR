import torch 
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class BiLSTM(nn.Module):
    def __init__(self, args, input_size=2048, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.args = args
        self.input_size = input_size
        # self.hidden_size = args.hidden_size
        # self.num_layers = args.num_layers
        # self.num_bilstm_blocks = args.num_bilstm_blocks
        
        self.hidden_size = 1024
        self.num_layers = 1
        self.num_bilstm_blocks = 2
        self.bilstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.attention = SelfAttention()
    def forward(self, x):
        for _ in range(self.num_bilstm_blocks):
            x, _ = self.bilstm(x)
            
        x = self.attention(x)     
        return x # (N, 25, 64)
    
# class BiLSTM(nn.Module):
#     def __init__(self, args, input_size=2048, hidden_dim1=512, hidden_dim2=32, output_dim=64):
#         super(BiLSTM, self).__init__()
#         self.args = args
#         self.bilstm1 = nn.LSTM(input_size, hidden_dim1, batch_first=True, bidirectional=True, num_layers=self.args.num_layers)
#         self.bilstm2 = nn.LSTM(hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=True, num_layers=self.args.num_layers)
#         self.attention = nn.Linear(hidden_dim2 * 2, output_dim)

#     def forward(self, x):
#         x, _ = self.bilstm1(x)  
#         x, _ = self.bilstm2(x)  
        
#         x = x[:, -1, :]
#         x = F.normalize(x)
#         return x 
    
# model = BiLSTM(None)
# x = torch.randn(48, 25, 2048)
# x = model(x)
# print(x.shape)
