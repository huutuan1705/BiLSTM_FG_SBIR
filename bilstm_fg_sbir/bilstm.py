import torch 
import torch.nn as nn

from attention import AttentionSequence

class BiLSTM(nn.Module):
    def __init__(self, input_size, num_layers, output_size, hidden_size = 512, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        
        if bidirectional:
            self.fc = nn.Linear(hidden_size*2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
            
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)

        attention = AttentionSequence(input_size=x.shape[1], hidden_layer=x.shape[1])
        output, _ = attention(x)
        
        return output


# x = torch.randn(25, 2048)
# model = BiLSTM(input_size=2048, num_layers=1, output_size=64, bidirectional=True)
# output = model(x)
# output = output
# print(output.shape) # [1, 64]