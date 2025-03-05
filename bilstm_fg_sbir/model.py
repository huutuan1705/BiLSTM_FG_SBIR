import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from backbones import VGG16, ResNet50, InceptionV3
from bilstm import BiLSTM
from attention import Attention_global, Linear_global, SelfAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM_FGSBIR_Model(nn.Module):
    def __init__(self, args):
        super(BiLSTM_FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(args.backbone_name + "(args)")
        self.sketch_embedding_network = eval(args.backbone_name + "(args)")
        self.loss = nn.TripletMarginLoss(margin=args.margin)        
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.sketch_train_params = self.sketch_embedding_network.parameters()
        self.args = args
        
        self.sample_embedding_network.fix_weights()
        self.sketch_embedding_network.fix_weights()
            
        self.bilstm_network = BiLSTM(args=args, input_size=2048).to(device)
        # self.bilstm_network.apply(init_weights)
        self.bilstm_params = self.bilstm_network.parameters()
        
        self.attention = SelfAttention(args)
        self.attention.fix_weights()
        
        self.sketch_attention = Attention_global()
        self.sketch_attention.fix_weights()
        
        self.linear = Linear_global(feature_num=self.args.output_size)
        self.linear.fix_weights()
        
        self.sketch_linear = Linear_global(feature_num=self.args.output_size)
        self.sketch_linear.fix_weights()
        
        self.optimizer = optim.Adam([
            {'params': self.bilstm_network.parameters(), 'lr': args.learning_rate},
        ])
        
    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        loss = 0
        for idx in range(len(batch['sketch_imgs'])):
            # print("batch['sketch_imgs'][idx].shape: ", batch['sketch_imgs'][idx].shape) # (N, 3, 299, 299)
            # print("batch['positive_img'][idx].shape: ", batch['positive_img'][idx].shape) # (3, 299, 299)
            sketch_feature = self.bilstm_network(self.attention(
                self.sample_embedding_network(batch['sketch_imgs'][idx].to(device))
            ))
            positive_feature = self.linear(self.attention(
                self.sample_embedding_network(batch['positive_img'][idx].unsqueeze(0).to(device))
            ))
            negative_feature = self.linear(self.attention(
                self.sample_embedding_network(batch['negative_img'][idx].unsqueeze(0).to(device))
            ))
            
            positive_feature = positive_feature.repeat(sketch_feature.shape[0], 1)
            negative_feature = negative_feature.repeat(sketch_feature.shape[0], 1)
            
            loss += self.loss(sketch_feature, positive_feature, negative_feature)
        
        loss.backward()
        self.optimizer.step()

        return loss.item() 
    
    def test_forward(self, batch):            
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        positive_feature = self.linear(self.attention(positive_feature))
        
        sketch_feature = self.attention(
            self.sample_embedding_network(batch['sketch_imgs'].squeeze(0).to(device))) # (25, 2048)
        
        return sketch_feature.cpu(), positive_feature.cpu()
    
    def evaluate(self, dataloader_test):
        self.eval()
        sketch_array_tests = []
        sketch_names = []
        image_array_tests = []
        image_names = []
        
        for idx, batch in enumerate(tqdm(dataloader_test)):
            sketch_features_all = torch.FloatTensor().to(device)
            for data_sketch in batch['sketch_imgs']:
                print(data_sketch.shape)
                sketch_feature = self.attention(
                    self.sample_embedding_network(data_sketch.to(device))
                )
                print("sketch_feature.shape: ", sketch_feature.shape)
                sketch_features_all = torch.cat((sketch_features_all, sketch_feature.detach()))\
            
            print("sketch_feature_ALL.shape: ", sketch_features_all.shape)
            
            sketch_array_tests.append(sketch_features_all.cpu())
        
        print("len(sketch_array_tests): ", len(sketch_array_tests))
            
            
            
        