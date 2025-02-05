import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, num_layers, output_size, hidden_size = 512, bidirectional=False):
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
        x = x.squeeze(0)
        return x

# x = torch.randn(1, 25, 2048)
# model = BiLSTM(input_size=2048, num_layers=1, output_size=64, bidirectional=True)
# output = model(x)
# output = output.squeeze(0)
# print(output.shape) --> output: [25, 64]