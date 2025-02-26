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
        # print("len(batch['sketch_imgs']): ", len(batch['sketch_imgs'])) #25
        for idx in range(len(batch['sketch_imgs'])):
            sketch_seq_feature = self.bilstm_network(self.attention(
                self.sample_embedding_network(batch['sketch_imgs'][idx].to(device))))
            # print(f'sketch_seq_feature: {sketch_seq_feature.shape}') # (1, 64)
            positive_feature = self.linear(self.attention(
                self.sample_embedding_network(batch['positive_img'][idx].unsqueeze(0).to(device))))
            # print(f'positive_feature: {positive_feature.shape}') # (1, 64)
            negative_feature = self.linear(self.attention(
                self.sample_embedding_network(batch['negative_img'][idx].unsqueeze(0).to(device))))
            # print(f'negative_feature: {negative_feature.shape}')
            positive_feature = positive_feature.repeat(sketch_seq_feature.shape[0], 1)
            negative_feature = negative_feature.repeat(sketch_seq_feature.shape[0], 1)
            
            loss += self.loss(sketch_seq_feature, positive_feature, negative_feature) 

        loss.backward()
        self.optimizer.step()
        return loss.item()

    
    def evaluate(self, dataloader_test):
        self.eval()
        sketch_array_tests = []
        sketch_names = []
        image_array_tests = torch.FloatTensor().to(device)
        image_names = []
        
        for idx, sampled_batch in enumerate(tqdm(dataloader_test)):
            sketch_feature_ALL = torch.FloatTensor().to(device)
            
            for data_sketch in sampled_batch['sketch_imgs']: 
                sketch_feature = self.attention(self.sample_embedding_network(data_sketch.to(device)))
                sketch_feature_ALL = torch.cat((sketch_feature_ALL, sketch_feature.detach()))
            
            sketch_names.extend(sampled_batch['sketch_path'])
            sketch_array_tests.append(sketch_feature_ALL.cpu())
            
            # print("sampled_batch['positive_path']: ", sampled_batch['positive_path'])
            if sampled_batch['positive_sample'][0] not in image_names:
                rgb_feature = self.linear(self.attention(
                    self.sample_embedding_network(sampled_batch['positive_img'].to(device))))
                image_array_tests = torch.cat((image_array_tests, rgb_feature.detach()))
                image_names.extend(sampled_batch['positive_sample'])
        # print("sketch_array_tests shape 2: ", sketch_array_tests.shape)
        
        sketch_steps = len(sketch_array_tests[0])

        avererage_area = []
        avererage_area_percentile = []
        
        rank_all = torch.zeros(len(sketch_array_tests), sketch_steps)
        rank_all_percentile = torch.zeros(len(sketch_array_tests), sketch_steps)
        
        print("sketch_array_tests shape: ", len(sketch_array_tests)) # 232
        for i_batch, sampled_batch in enumerate(sketch_array_tests):
            mean_rank = []
            mean_rank_percentile = []
            sketch_name = sketch_names[i_batch]
            print(f'sketch_name: {sketch_name}')
            
            
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = image_names.index(sketch_query_name)
            
            print("sampled_batch shape: ", sampled_batch.shape) # (25, 2048)
            for i_sketch in range(sampled_batch.shape[0]):
                print("sampled_batch[i_sketch] shape: ", sampled_batch[i_sketch].shape) # (2048, )
                sketch_feature = self.bilstm_network(sampled_batch[i_sketch].unsqueeze(0).to(device))
                
                print("Sketch feature shape: ", sketch_feature.shape) # (1, 2048)
                print("image_array_tests[position_query]: ", image_array_tests[position_query].shape) #(2048, )
                print("image_array_tests shape: ", image_array_tests.shape) # (100, 2048)
                target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0).to(device), image_array_tests[position_query].unsqueeze(0).to(device))
                distance = F.pairwise_distance(sketch_feature.unsqueeze(0).to(device), image_array_tests.to(device))
                print(f'distance: {len(distance)}')
                
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()
                rank_all_percentile[i_batch, i_sketch] = (len(distance[0]) - rank_all[i_batch, i_sketch]) / (len(distance[0]) - 1)
                
                mean_rank.append(1/rank_all[i_batch, i_sketch].item() if rank_all[i_batch, i_sketch].item()!=0 else 1)
                mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item() if rank_all_percentile[i_batch, i_sketch].item()!=0 else 1)
            
            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            
            print("len(mean_rank_percentile): ", len(mean_rank_percentile))
            print("np.sum(mean_rank_percentile): ", np.sum(mean_rank_percentile))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
        
        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        
        meanMB = np.mean(avererage_area)
        meanMA = np.mean(avererage_area_percentile)
        
        return top1_accuracy, top5_accuracy, top10_accuracy, meanMA, meanMB