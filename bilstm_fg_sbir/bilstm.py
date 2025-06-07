import torch 
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    def __init__(self, args, input_dim=2048, hidden_dim=512, output_dim=64, num_layers=1, dropout=0.2):
        super(BiLSTM, self).__init__()

        self.bi_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Attention layer
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)

        # Output projection to final embedding space
        self.output_fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, sketch_seq):
        """
        sketch_seq: Tensor of shape (batch_size, seq_len, input_dim)
        """
        # Bi-LSTM output
        lstm_out, _ = self.bi_lstm(sketch_seq)  # shape: (batch, seq_len, hidden_dim*2)

        # Compute attention scores
        attn_scores = self.attention_fc(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # normalize over sequence length

        context_vector = attn_weights * lstm_out  # (batch, seq_len, hidden_dim*2)

        # Final projection
        output = self.output_fc(context_vector)  # (batch, output_dim)
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
    
# model = BiLSTM()
# x = torch.randn(3, 25, 2048)
# x = model(x)
# print(x.shape) #(3, 25, 64)
