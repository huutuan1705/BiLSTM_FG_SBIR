import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention_global(nn.Module):
    def __init__(self):
        super(Attention_global, self).__init__()
        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        self.net = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=1),
                                 nn.BatchNorm2d(1024),
                                 nn.ReLU(),
                                 nn.Conv2d(1024, 1, kernel_size=1))
    
    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False
              
    def forward(self, backbone_tensor):
        backbone_tensor_1 = self.net(backbone_tensor)
        backbone_tensor_1 = backbone_tensor_1.view(backbone_tensor_1.size(0), -1)
        backbone_tensor_1 = nn.Softmax(dim=1)(backbone_tensor_1)
        backbone_tensor_1 = backbone_tensor_1.view(backbone_tensor_1.size(0), 1, backbone_tensor.size(2), backbone_tensor.size(3))
        fatt = backbone_tensor*backbone_tensor_1
        fatt1 = backbone_tensor +fatt
        fatt1 = self.pool_method(fatt1).view(-1, 2048)
        return  F.normalize(fatt1)

class AttentionSequence(nn.Module):
    def __init__(self):
        super(AttentionSequence, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.output_layer = nn.Linear(2048, 64)  # Biến đổi đầu ra cuối cùng về (N, 25, 64)
    
    def forward(self, x):
        batch_size = x.size(0)
        x_reshaped = x.view(-1, x.size(-1))
        attention_weights = self.attention_net(x_reshaped)
        attention_weights = attention_weights.view(batch_size, -1)
        attention_weights = nn.Softmax(dim=1)(attention_weights)
        attention_weights = attention_weights.unsqueeze(-1)
        weighted = x * attention_weights
        enhanced = x + weighted
        
        enhanced_flat = enhanced.view(-1, enhanced.size(-1))
        
        projected = self.output_layer(enhanced_flat)
        # print("projected shape: ", projected.shape)
        output = projected.view(batch_size, 25, 64)
        
        # Normalize output
        output = F.normalize(projected, dim=-1)
        
        return output
    
class Linear_global(nn.Module):
    def __init__(self, feature_num):
        super(Linear_global, self).__init__()
        self.head_layer = nn.Linear(2048, feature_num)
    
    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False
            
    def forward(self, x):
        return F.normalize(self.head_layer(x))

# input_tensor = torch.randn(48, 2048, 8, 8)
# model = Attention_global()
# output = model(input_tensor)

x = torch.randn(1, 25, 2048)  
model = AttentionSequence()
output = model(x)
print(output.shape)
