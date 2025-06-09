import torch 
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    def __init__(self, args, input_dim=2048, hidden_dim=512, output_dim=64, dropout=0.1):
        super(BiLSTM, self).__init__()

        self.bi_lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.bi_lstm2 = nn.LSTM(
            input_size=hidden_dim*2,
            hidden_size=32,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Attention layer
        self.attention_fc = nn.Linear(64, 1)


    def forward(self, sketch_seq):
        """
        sketch_seq: Tensor of shape (batch_size, seq_len, input_dim)
        """
        # Bi-LSTM output
        lstm_out1, _ = self.bi_lstm1(sketch_seq)  # shape: (batch, seq_len, hidden_dim*2)
        lstm_out2, _ = self.bi_lstm2(lstm_out1)

        # Compute attention scores
        attn_scores = self.attention_fc(lstm_out2)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # normalize over sequence length

        context_vector = attn_weights * lstm_out2  # (batch, seq_len, hidden_dim*2)

        return F.normalize(context_vector)

    
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
