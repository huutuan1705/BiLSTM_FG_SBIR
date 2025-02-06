import torch
import torch.nn as nn

class AttentionImage(nn.Module):
    def __init__(self, input_size, hidden_layer=2048):
        super(AttentionImage, self).__init__()  
        self.input_size = input_size      
        self.attn_hidden_layer = hidden_layer

        self.net = nn.Sequential(
            nn.Conv2d(self.input_size, self.attn_hidden_layer, kernel_size=1), 
            nn.BatchNorm2d(self.attn_hidden_layer), 
            nn.ReLU(),
            nn.Conv2d(self.attn_hidden_layer, 1, kernel_size=1)
        )
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        attn_mask = self.net(x)  # (batch_size, 1, H, W)
        attn_mask = attn_mask.view(attn_mask.size(0), -1)  # reshape (batch_size, H*W)
        attn_mask = self.softmax(attn_mask)  # Softmax
        attn_mask = attn_mask.view(x.size(0), 1, x.size(2), x.size(3))  # reshape (batch_size, 1, H, W)

        x_attn = x * attn_mask  # attention
        x = x + x_attn  # Residual connection

        return x, attn_mask
    
class AttentionSequence(nn.Module):
    def __init__(self, input_size, hidden_layer=128):
        super(AttentionSequence, self).__init__()
        self.input_size = input_size
        self.hidden_layer = hidden_layer

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=self.hidden_layer, kernel_size=1),  
            nn.BatchNorm1d(self.hidden_layer),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.hidden_layer, out_channels=1, kernel_size=1)  
        )
        self.softmax = nn.Softmax(dim=2)  # Softmax for N dims (sequence_length)

    def forward(self, x):
        """
        x: Tensor shape (batch_size, N, d) với N=25, d=64
        """
        if x.dim() == 2:  # If input don't have batch dimension
            x = x.unsqueeze(0)  # add batch dimension => (1, N, d)

        x = x.permute(0, 2, 1)  # reshape (batch_size, d, N)

        attn_mask = self.net(x)  # (batch_size, 1, N)
        attn_mask = self.softmax(attn_mask)  # Softmax normalize attention weights

        x_weighted = torch.bmm(attn_mask, x.permute(0, 2, 1))  # (batch_size, 1, N) @ (batch_size, N, d) => (batch_size, 1, d)
        x_weighted = x_weighted.squeeze(1)  # (batch_size, d)

        # return (batch_size, d) and attention weights (batch_size, 1, N)
        return x_weighted, attn_mask  

# input_tensor = torch.randn(25, 64)
# model = AttentionSequence(input_size=64)
# output, attn_mask = model(input_tensor)

# print("Output shape:", output.shape)  # (1, 64)
# print("Attention mask shape:", attn_mask.shape)  # (1, 1, 25)
