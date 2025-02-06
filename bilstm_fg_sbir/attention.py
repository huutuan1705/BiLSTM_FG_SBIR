import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_size, hidden_layer=2048):
        super(Attention, self).__init__()  
        self.input_size = input_size      
        self.attn_hidden_layer = hidden_layer

        self.net = nn.Sequential(
            nn.Conv2d(self.input_size, self.attn_hidden_layer, kernel_size=1), 
            nn.BatchNorm2d(self.attn_hidden_layer),  # Sửa lỗi BatchNorm1d -> BatchNorm2d
            nn.ReLU(),
            nn.Conv2d(self.attn_hidden_layer, 1, kernel_size=1)
        )
        self.softmax = nn.Softmax(dim=1)  # Định nghĩa softmax trong __init__

    def forward(self, x):
        attn_mask = self.net(x)  # (batch_size, 1, H, W)
        attn_mask = attn_mask.view(attn_mask.size(0), -1)  # Chuyển thành (batch_size, H*W)
        attn_mask = self.softmax(attn_mask)  # Softmax theo không gian
        attn_mask = attn_mask.view(x.size(0), 1, x.size(2), x.size(3))  # Chuyển lại về (batch_size, 1, H, W)

        x_attn = x * attn_mask  # Áp dụng attention
        x = x + x_attn  # Residual connection

        return x, attn_mask

input_tensor = torch.randn(1, 2048, 8, 8)
model = Attention(input_size=2048, hidden_layer=2048)
output, attn_mask = model(input_tensor)

print("Output shape:", output.shape)  # (1, 2048, 8, 8)
print("Attention mask shape:", attn_mask.shape)  # (1, 1, 8, 8)
