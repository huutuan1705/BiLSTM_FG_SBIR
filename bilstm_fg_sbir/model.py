import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from backbones import VGG16, ResNet50, InceptionV3
from bilstm import BiLSTM
from attention import Attention_global, Linear_global

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
        
        self.attention = Attention_global()
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
        print("len(batch['sketch_imgs']): ", len(batch['sketch_imgs']))
        for idx in range(len(batch['sketch_imgs'])):
            sketch_seq_feature = self.bilstm_network(self.attention(
                self.sample_embedding_network(batch['sketch_imgs'][idx].to(device))))
            positive_feature = self.linear(self.attention(
                self.sample_embedding_network(batch['positive_img'][idx].unsqueeze(0).to(device))))
            print(f'positive_feature: {positive_feature.shape}')
            negative_feature = self.linear(self.attention(
                self.sample_embedding_network(batch['negative_img'][idx].unsqueeze(0).to(device))))
            # print(f'negative_feature: {negative_feature.shape}')
            positive_feature = positive_feature.repeat(sketch_seq_feature.shape[0], 1)
            negative_feature = negative_feature.repeat(sketch_seq_feature.shape[0], 1)
            
            loss += self.loss(sketch_seq_feature, positive_feature, negative_feature) 

        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def test_forward(self, batch):            #  this is being called only during evaluation
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        positive_feature = self.linear(self.attention(positive_feature))
        
        # print("positive_feature shape: ", positive_feature.shape)
        sketch_feature =  self.attention(self.sample_embedding_network(batch['sketch_seq'].squeeze(0).to(device)))
        return sketch_feature.cpu(), positive_feature.cpu()
    
    def evaluate(self, dataloader_test):
        self.eval()
        sketch_array_tests = []
        sketch_names = []
        image_array_tests = []
        image_names = []
        
        for idx, batch in enumerate(tqdm(dataloader_test)):
            sketch_feature, positive_feature = self.test_forward(batch)
            sketch_array_tests.append(sketch_feature)
            sketch_names.append(batch['sketch_path'])
            
            for i_num, positive_name in enumerate(batch['positive_path']): 
                if positive_name not in image_names:
                    image_names.append(batch['positive_sample'][i_num])
                    image_array_tests.append(positive_feature[i_num])
                
        sketch_array_tests = torch.stack(sketch_array_tests)
        image_array_tests = torch.stack(image_array_tests)
        
        # print("sketch_array_tests shape 2: ", sketch_array_tests.shape)
        
        sketch_steps = len(sketch_array_tests[0])

        avererage_area = []
        avererage_area_percentile = []
        
        rank_all = torch.zeros(len(sketch_array_tests), sketch_steps)
        rank_all_percentile = torch.zeros(len(sketch_array_tests), sketch_steps)
        
        # print("rank_all_percentile shape: ", rank_all_percentile.shape)
        for i_batch, sanpled_batch in enumerate(sketch_array_tests):
            mean_rank = []
            mean_rank_percentile = []
            sketch_name = sketch_names[i_batch][0]
            # print(f'sketch_name: {sketch_name}')
            
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = image_names.index(sketch_query_name)
            
            # print("sanpled_batch shape: ", sanpled_batch.shape) # (1, 25, 2048)
            for i_sketch in range(sanpled_batch.shape[0]):
                sketch_features = self.bilstm_network(sanpled_batch[:i_sketch+1].to(device))
                target_distance = F.pairwise_distance(sketch_feature[-1].unsqueeze(0).to(device), self.Image_Array_Test[position_query].unsqueeze(0).to(device))
                distance = F.pairwise_distance(sketch_feature[-1].unsqueeze(0).to(device), self.Image_Array_Test.to(device))
                # print(f'distance: {len(distance)}')
                
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()
                rank_all_percentile[i_batch, i_sketch] = (len(distance[0]) - rank_all[i_batch, i_sketch]) / (len(distance[0]) - 1)
                
                mean_rank.append(1/rank_all[i_batch, i_sketch].item() if rank_all[i_batch, i_sketch].item()!=0 else 1)
                mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item() if rank_all_percentile[i_batch, i_sketch].item()!=0 else 1)
            
            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            # print("len(mean_rank_percentile): ", len(mean_rank_percentile))
            # print("np.sum(mean_rank_percentile): ", np.sum(mean_rank_percentile))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
        
        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        
        meanMB = np.mean(avererage_area)
        meanMA = np.mean(avererage_area_percentile)
        
        return top1_accuracy, top5_accuracy, top10_accuracy, meanMA, meanMB