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
        self.args = args
        
        self.sample_embedding_network.fix_weights()
        self.sketch_embedding_network.fix_weights()
            
        self.bilstm_network = BiLSTM(args=args, input_size=2048).to(device)
        # self.bilstm_network.apply(init_weights)
        self.bilstm_params = self.bilstm_network.parameters()
        
        self.attention = SelfAttention(args)
        self.attention.fix_weights()
        self.sketch_attention = SelfAttention(args)
        self.sketch_attention.fix_weights()
        
        self.linear = Linear_global(feature_num=self.args.output_size)
        self.linear.fix_weights()
        
        
        self.optimizer = optim.Adagrad([
            {'params': self.bilstm_network.parameters(), 'lr': args.learning_rate},
        ])
        
    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        positive_features = self.linear(self.attention(
            self.sample_embedding_network(batch['positive_img'].to(device))
        )) # (N, 64)
        negative_features = self.linear(self.attention(
            self.sample_embedding_network(batch['negative_img'].to(device))
        )) # (N, 64)
        
        loss = 0
        # print("len(batch['sketch_imgs']): ", len(batch['sketch_imgs'])) # 64
        for i in range(len(batch['sketch_imgs'])):
            sketch_features = self.sketch_attention(
                self.sketch_embedding_network(batch['sketch_imgs'][i].to(device))) # (25, 2048)
            
            # print("sketch_features[:i_sketch+1].shape: ", sketch_features[:i_sketch+1].shape)
            sketch_feature = self.bilstm_network(sketch_features)
            # print("positive_features[i].shape: ", positive_features[i].shape) # (64, )
            positive_feature = positive_features[i]
            negative_feature = negative_features[i]
            
            loss += self.loss(F.normalize(sketch_feature), positive_feature.unsqueeze(0), negative_feature.unsqueeze(0))
        
        loss.backward()
        self.optimizer.step()

        return loss.item() 
    
    
    def evaluate(self, dataloader_test):
        self.eval()
        sketch_array_tests = []
        sketch_names = []
        image_array_tests = torch.FloatTensor().to(device)
        image_names = []
        
        for idx, batch in enumerate(tqdm(dataloader_test)):
            sketch_features_all = torch.FloatTensor().to(device)
            for data_sketch in batch['sketch_imgs']:
                # print(data_sketch.shape) # (1, 3, 299, 299)
                sketch_feature = self.sketch_attention(
                    self.sketch_embedding_network(data_sketch.to(device))
                )
                # print("sketch_feature.shape: ", sketch_feature.shape) #(1, 2048)
                sketch_features_all = torch.cat((sketch_features_all, sketch_feature.detach()))
            
            # print("sketch_feature_ALL.shape: ", sketch_features_all.shape) # (25, 2048)           
            sketch_array_tests.append(sketch_features_all.cpu())
            sketch_names.extend(batch['sketch_path'])
            
            if batch['positive_path'][0] not in image_names:
                positive_feature = self.linear(self.attention(
                    self.sample_embedding_network(batch['positive_img'].to(device))))
                image_array_tests = torch.cat((image_array_tests, positive_feature))
                image_names.extend(batch['positive_path'])
                
        num_steps = len(sketch_array_tests[0])
        avererage_area = []
        avererage_area_percentile = []
                
        rank_all = torch.zeros(len(sketch_array_tests), num_steps)
        rank_all_percentile = torch.zeros(len(sketch_array_tests), num_steps)
                
        for i_batch, sampled_batch in enumerate(sketch_array_tests):
            mean_rank = []
            mean_rank_percentile = []
            sketch_name = sketch_names[i_batch]
            
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = image_names.index(sketch_query_name)
            sketch_features = self.bilstm_network(sampled_batch.to(device))
            
            for i_sketch in range(sampled_batch.shape[0]):
                # print("sampled_batch[:i_sketch+1].shape: ", sampled_batch[:i_sketch+1].shape)
                sketch_feature = sketch_features[i_sketch]
                target_distance = F.pairwise_distance(F.normalize(sketch_feature).to(device), image_array_tests[position_query].to(device))
                distance = F.pairwise_distance(F.normalize(sketch_feature).to(device), image_array_tests.to(device))
                
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()

                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)
                if rank_all[i_batch, i_sketch].item() == 0:
                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
                        #1/(rank)
                    mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item())
            
            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
        
        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        
        meanMA = np.mean(avererage_area_percentile)
        meanMB = np.mean(avererage_area)
        
        return top1_accuracy, top5_accuracy, top10_accuracy, meanMA, meanMB