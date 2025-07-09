import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.pool_method =  nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(2048)
        self.mha = nn.MultiheadAttention(2048, num_heads=args.num_heads, batch_first=True)
        # self.mha = nn.MultiheadAttention(2048, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        identify = x
        bs, c, h, w = x.shape
        x_att = x.reshape(bs, c, h*w).transpose(1, 2)
        x_att = self.norm(x_att)
        att_out, _  = self.mha(x_att, x_att, x_att)
        att_out = self.dropout(att_out)
        att_out = att_out.transpose(1, 2).reshape(bs, c, h, w)
        
        output = identify * att_out + identify
        output = self.pool_method(output).view(-1, 2048)
        return F.normalize(output)
    
    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False
    
    
class Linear_global(nn.Module):
    def __init__(self, feature_num):
        super(Linear_global, self).__init__()
        self.head_layer = nn.Linear(2048, feature_num)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(x)
        return F.normalize(self.head_layer(x))
    
    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False