import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbone import InceptionV3
from src.attention import Linear_global, SelfAttention
from src.bilstm import BiLSTM

class BiLSTM_SBIR(nn.Module):
    def __init__(self, args):
        super(BiLSTM_SBIR, self).__init__()
        
        self.args = args
        self.sample_embedding_network = InceptionV3(args=args)
        self.attention = SelfAttention(args)
        self.linear = Linear_global(feature_num=self.args.output_size)
        
        self.sketch_embedding_network = InceptionV3(args=args)
        self.sketch_attention = SelfAttention(args)
        self.sketch_linear = Linear_global(feature_num=self.args.output_size)

        self.sample_embedding_network.fix_weights()
        self.attention.fix_weights()
        self.linear.fix_weights()
        self.sketch_embedding_network.fix_weights()
        self.sketch_attention.fix_weights()
        self.sketch_linear.fix_weights()
        
        self.bilstm = BiLSTM()
        
    def forward(self, batch):
        sketch_imgs = batch['sketch_imgs']
        positive_img = batch['positive_img']
        negative_img = batch['negative_img']
        
        positive_feature = self.linear(self.attention(self.sample_embedding_network(positive_img)))
        negative_feature = self.linear(self.attention(self.sample_embedding_network(negative_img)))
        sketch_features = self.sketch_attention(self.sketch_embedding_network(sketch_imgs))
        
        sketch_features = self.sketch_linear(self.bilstm(sketch_features))
        
        return positive_feature, negative_feature, sketch_features
    
    def test_forward(self, batch):
        sketch_imgs = batch['sketch_imgs']
        positive_img = batch['positive_img']
        
        positive_feature = self.linear(self.attention(self.sample_embedding_network(positive_img)))
        sketch_features = self.sketch_attention(self.sketch_embedding_network(sketch_imgs))
        
        sketch_features = self.sketch_linear(self.bilstm(sketch_features))
        
        return positive_feature, sketch_features