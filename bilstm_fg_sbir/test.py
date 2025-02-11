import torch.nn as nn
from torch import optim
import torch
import time
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Model(nn.Module):
    def __init__(self, opt):
        super(FGSBIR_Model, self).__init__()
        # InceptionV3
        self.backbone_network = InceptionV3_Network()
        self.backbone_network.load_state_dict(torch.load("/kaggle/input/mode1_chairv2/pytorch/default/1/ChairV2_2024_12_16_12.06.21_inceptionv3_uoc.pth",
                                                         map_location= "cuda", 
                                                         weights_only=True))
        self.backbone_network.to(device)
        self.backbone_network.fixed_param()
        self.backbone_network.eval()
        
        
        self.attn_network = Attention()
        self.attn_network.load_state_dict(torch.load("/kaggle/input/mode1_chairv2/pytorch/default/1/ChairV2_2024_12_16_12.06.21_attention_uoc.pth", 
                                                     map_location= "cuda", 
                                                     weights_only=True))
        self.attn_network.to(device)
        self.attn_network.fixed_param()
        self.attn_network.eval()

        # Linear
        self.linear_network = Linear()
        self.linear_network.load_state_dict(torch.load("/kaggle/input/mode1_chairv2/pytorch/default/1/ChairV2_2024_12_16_12.06.21_linear_uoc.pth", 
                                                       map_location= "cuda", 
                                                       weights_only=True))
        self.linear_network.to(device)
        self.linear_network.fixed_param()
        self.linear_network.eval()

        # LSTM
        self.lstm_network = LSTM()
        self.lstm_network.train()
        self.lstm_network.to(device)
        self.lstm_train_params = self.lstm_network.parameters()

        self.optimizer = optim.Adam([
            {'params': self.lstm_train_params, 'lr': 0.0005 }])
        
        self.loss = nn.TripletMarginLoss(margin=0.3)
        self.opt = opt
    
    def train_model(self, batch):
        self.backbone_network.eval()
        self.attn_network.eval()
        self.linear_network.eval()
        self.lstm_network.train()

        loss = 0

        for idx in range(len(batch['sketch_seq'])):
            sketch_seq_feature = self.lstm_network(self.attn_network(
                self.backbone_network(batch['sketch_seq'][idx].to(device))))
            # print(f'sketch_seq_feature: {sketch_seq_feature.shape}')
            positive_feature = self.linear_network(self.attn_network(
                self.backbone_network(batch['positive_img'][idx].unsqueeze(0).to(device))))
            # print(f'positive_feature: {positive_feature.shape}')
            negative_feature = self.linear_network(self.attn_network(
                self.backbone_network(batch['negative_img'][idx].unsqueeze(0).to(device))))
            # print(f'negative_feature: {negative_feature.shape}')
            positive_feature = positive_feature.repeat(sketch_seq_feature.shape[0], 1)
            negative_feature = negative_feature.repeat(sketch_seq_feature.shape[0], 1)
            
            loss += self.loss(sketch_seq_feature, positive_feature, negative_feature)      

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate_NN(self, dataloader):
        self.backbone_network.eval()
        self.attn_network.eval()
        self.lstm_network.eval()
        self.linear_network.eval()
    
        self.Sketch_Array_Test = []
        self.Sketch_Name = []
        self.Image_Array_Test = []
        self.Image_Name = []
        for idx, batch in enumerate(dataloader):
            sketch_feature = self.attn_network(
                self.backbone_network(batch['sketch_seq'].squeeze(0).to(self.opt.device)))
            positive_feature = self.linear_network(self.attn_network(
                self.backbone_network(batch['positive_img'].to(self.opt.device))))
                
            self.Sketch_Array_Test.append(sketch_feature)
            self.Sketch_Name.append(batch['sketch_path'])
                
                # self.Image_Array_Test.append(positive_feature)
            for i_num, positive_name in enumerate(batch['positive_path']): 
                if positive_name not in self.Image_Name: 
                    self.Image_Name.append(batch['positive_path'][i_num])
                    self.Image_Array_Test.append(positive_feature[i_num])
                        
                        
                        
                
        self.Sketch_Array_Test = torch.stack(self.Sketch_Array_Test)
        # print(f'Sketch_Array_Test: {self.Sketch_Array_Test.shape}')
        # print(f'Sketch_Name: {len(self.Sketch_Name)}')
        self.Image_Array_Test = torch.stack(self.Image_Array_Test)
        # print(f'Image_Array_Test: {self.Image_Array_Test.shape}')
        # print(f'Image_Name: {len(self.Image_Name)}')
         
        num_of_Sketch_Step = len(self.Sketch_Array_Test[0])

        # print(f'num_of_Sketch_Step: {num_of_Sketch_Step}')
        avererage_area = []
        avererage_area_percentile = []
         
        rank_all = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)
        # print(f'rank_all: {rank_all.shape}')
        rank_all_percentile = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)
        # print(f'rank_all_percentile: {rank_all_percentile.shape}')
            # sketch_range = torch.Tensor(sketch_range)
         
        for i_batch, sanpled_batch in enumerate(self.Sketch_Array_Test):
            mean_rank = []
            mean_rank_percentile = []
            sketch_name = self.Sketch_Name[i_batch][0]
            # print(f'sketch_name: {sketch_name}')
            sketch_query_name = ''.join(sketch_name.split('/')[-1].split('')[:-1])
            # print(f'sketch_query_name: {sketch_query_name}')
            position_query = self.Image_Name.index(sketch_query_name)
            # print(f'position_query: {position_query}')
                
            for i_sketch in range(sanpled_batch.shape[0]):
                sketch_feature = self.lstm_network(sanpled_batch[:i_sketch+1].to(device))
                target_distance = F.pairwise_distance(sketch_feature[-1].unsqueeze(0).to(device), self.Image_Array_Test[position_query].unsqueeze(0).to(device))
                distance = F.pairwise_distance(sketch_feature[-1].unsqueeze(0).to(device), self.Image_Array_Test.to(device))
                # print(f'distance: {len(distance)}')
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()

                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)
                if rank_all[i_batch, i_sketch].item() == 0:
                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
                        #1/(rank)
                    mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item())
                        #rank_percentile
            # print(rank_all[i_batch])
            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))

        # print(rank_all)
        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        top20_accuracy = rank_all[:, -1].le(20).sum().numpy() / rank_all.shape[0]
        top50_accuracy = rank_all[:, -1].le(50).sum().numpy() / rank_all.shape[0]
            #A@1 A@5 A%10
        meanMB = np.mean(avererage_area)
        meanMA = np.mean(avererage_area_percentile)

    
        return top1_accuracy, top5_accuracy, top10_accuracy, top20_accuracy, top50_accuracy, meanMB, meanMA