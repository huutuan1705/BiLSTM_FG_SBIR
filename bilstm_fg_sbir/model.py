import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from backbones import VGG16, ResNet50, InceptionV3
from bilstm import BiLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM_FGSBIR_Model(nn.Module):
    def __init__(self, args):
        super(BiLSTM_FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(args.backbone_name + "(args)")
        self.loss = nn.TripletMarginLoss(margin=0.3)
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.optimizer = optim.Adam(self.sample_train_params, args.learning_rate)
        self.args = args
    
    def test_forward(self, batch):            #  this is being called only during evaluation
        sketch_feature = self.sample_embedding_network(batch['sketch_imgs'].to(device))
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        return sketch_feature.cpu(), positive_feature.cpu()
    
    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        for param in self.sample_embedding_network.parameters():
            param.requires_grad = False
        
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        negative_feature = self.sample_embedding_network(batch['negative_img'].to(device))
        
        print(len(batch['sketch_imgs']))
        
        sketch_features = []
        for data_sketch in batch['sketch_imgs']:
            sketch_feature = self.sample_embedding_network(data_sketch.to(device))
            sketch_features.append(sketch_feature)
        
        print(sketch_features.shape)
        
        # Linear to get ouput (batch_size, 64)
        positive_linear = nn.Linear(positive_feature.shape[-1], self.args.output_size).to(device)
        negative_linear = nn.Linear(negative_feature.shape[-1], self.args.output_size).to(device)
        
        positive_feature = positive_linear(positive_feature)
        negative_feature = negative_linear(negative_feature)
        
        bilstm = BiLSTM(input_size=sketch_features.shape[-1], num_layers=self.args.num_layers, 
                        output_size=self.args.output_size).to(device)
        sketch_features = bilstm(sketch_features)
        
        loss = self.loss(sketch_features, positive_feature, negative_feature)
        loss.backward()
        self.optimizer.step()

        return loss.item() 
        
    def evaluate(self, dataloader_test):
        self.eval()
        
        Image_Feature_ALL = []
        Image_Name = []
        Sketch_Feature_ALL = []
        Sketch_Name = []
        
        for i_batch, sanpled_batch in enumerate(tqdm(dataloader_test)):
            sketch_feature, positive_feature= self.test_forward(sanpled_batch)
            Sketch_Feature_ALL.extend(sketch_feature)
            Sketch_Name.extend(sanpled_batch['sketch_path'])

            for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
                if positive_name not in Image_Name:
                    Image_Name.append(sanpled_batch['positive_path'][i_num])
                    Image_Feature_ALL.append(positive_feature[i_num])

        rank = torch.zeros(len(Sketch_Name))
        Image_Feature_ALL = torch.stack(Image_Feature_ALL)

        for num, sketch_feature in enumerate(Sketch_Feature_ALL):
            s_name = Sketch_Name[num]
            sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
            position_query = Image_Name.index(sketch_query_name)

            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                  Image_Feature_ALL[position_query].unsqueeze(0))

            rank[num] = distance.le(target_distance).sum()

        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top5 = rank.le(5).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]

        # print('Time to EValuate:{}'.format(time.time() - start_time))
        return top1, top5, top10
        
        