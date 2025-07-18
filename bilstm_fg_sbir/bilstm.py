import torch 
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    def __init__(self, input_size=2048):
        super(BiLSTM, self).__init__()
        
        self.bilstm1 = nn.LSTM(input_size=input_size, hidden_size=input_size//2, batch_first=True, bidirectional=True, dropout=0.2)
        self.bilstm2 = nn.LSTM(input_size=input_size, hidden_size=input_size//2, batch_first=True, bidirectional=True, dropout=0.2)
    
    def forward(self, x):
        identify = x
        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)
        
        output = identify + x
        return F.normalize(output)

    
# class BiLSTM(nn.Module):
#     def __init__(self, args, input_size=2048, hidden_dim1=512, hidden_dim2=32, output_dim=64):
#         super(BiLSTM, self).__init__()
#         self.args = args
#         self.bilstm1 = nn.LSTM(input_size, hidden_dim1, batch_first=True, bidirectional=True, num_layers=self.args.num_layers)
#         self.bilstm2 = nn.LSTM(hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=True, num_layers=self.args.num_layers)
        
#         # self.bilstm1 = nn.LSTM(input_size, hidden_dim1, batch_first=True, bidirectional=True, num_layers=2)
#         # self.bilstm2 = nn.LSTM(hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=True, num_layers=2)
#         # self.attention = nn.Linear(hidden_dim2 * 2, output_dim)

#     def forward(self, x):
#         x, _ = self.bilstm1(x)  
#         x, _ = self.bilstm2(x)  
        
#         # x = x[-1, :]
#         # x = F.normalize(x)
#         return x 
    
# model = BiLSTM(None)
# x = torch.randn(3, 25, 2048)
# x = model(x)
# print(x.shape) #(3, 25, 64)
