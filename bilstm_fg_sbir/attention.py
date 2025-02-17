import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention_global(nn.Module):
    def __init__(self):
        super(Attention_global, self).__init__()
        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        self.net = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(),
                                 nn.Conv2d(512, 1, kernel_size=1))
    
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

class Attention_sequence(nn.Module):
    def __init__(self):
        super(Attention_sequence, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2048, 512, kernel_size=1),  
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1, kernel_size=1)
        )

    def fix_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.transpose(1, 2)  # transpose (N, 2048, 25)
        
        # get attention weights
        attn_weights = self.net(x)  # (N, 1, 25)
        attn_weights = attn_weights.squeeze(1)  # (N, 25)
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)  # (N, 1, 25)
        # print(attn_weights)
        fatt = x * attn_weights  # (N, 2048, 25)
        fatt1 = x + fatt
        fatt1 = F.normalize(fatt1, dim=1)
        return fatt1.transpose(1, 2) 
    
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

# N, D = 4, 2048  # Batch size 4, Feature size 2048
# x = torch.randn(48, 25, 2048)  
# model = Attention_sequence()
# output = model(x)

# print(output.shape)
