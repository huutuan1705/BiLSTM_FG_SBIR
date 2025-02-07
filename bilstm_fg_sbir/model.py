import torch
import torch.nn as nn

from torch import optim
from tqdm import tqdm

from backbones import VGG16, ResNet50, InceptionV3
from bilstm import BiLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM_FGSBIR_Model(nn.Module):
    def __init__(self, args):
        super(BiLSTM_FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(args.backbone_name + "(args)")
        self.loss = nn.TripletMarginLoss(margin=0.2)
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.optimizer = optim.Adam(self.sample_train_params, args.learning_rate)
        self.args = args
        
    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        negative_feature = self.sample_embedding_network(batch['negative_img'].to(device))
        sketch_features = self.sample_embedding_network(batch['sketch_imgs'].to(device))
        
        bilstm = BiLSTM(input_size=sketch_features.shape[1], )
        
        