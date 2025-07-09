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