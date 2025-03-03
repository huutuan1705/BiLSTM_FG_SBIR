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
        identify = backbone_tensor
        backbone_tensor_1 = self.net(backbone_tensor)
        backbone_tensor_1 = backbone_tensor_1.view(backbone_tensor_1.size(0), -1)
        backbone_tensor_1 = nn.Softmax(dim=1)(backbone_tensor_1)
        backbone_tensor_1 = backbone_tensor_1.view(backbone_tensor_1.size(0), 1, backbone_tensor.size(2), backbone_tensor.size(3))
        fatt = identify*backbone_tensor_1
        fatt1 = identify +fatt
        fatt1 = self.pool_method(fatt1).view(-1, 2048)
        return  F.normalize(fatt1)

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.norm = nn.LayerNorm(2048)
        self.mha = nn.MultiheadAttention(2048, num_heads=4, batch_first=True)
        self.pool1d = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(2048, 64)
    
    def forward(self, x):
        x = self.norm(x)  
        att_out, _ = self.mha(x, x, x)  # (bs, 25, 2048)
        att_out = att_out.permute(0, 2, 1)
        att_out = self.pool1d(att_out).view(-1, 2048)
        att_out = F.normalize(att_out)
        return att_out
    
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

# x = torch.randn(48, 25, 2048)  
# model = SelfAttention()
# output = model(x)
# print(output.shape)
