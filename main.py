import os
import argparse
import torch
import torch.nn as nn
from src.model import BiLSTM_SBIR
from src.train import train_model

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Parameter:
        nn.init.kaiming_normal_(m.weight)
        
if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='GAT Fine-Grained SBIR model')
    parsers.add_argument('--dataset_name', type=str, default='ShoeV2')
    parsers.add_argument('--output_size', type=int, default=64)
    parsers.add_argument('--num_heads', type=int, default=8)
    parsers.add_argument('--root_dir', type=str, default='/kaggle/input/fg-sbir-dataset')
    parsers.add_argument('--pretrained_dir', type=str, default='/kaggle/input/chairv2_pretrained/pytorch/default/1')
    
    parsers.add_argument('--use_kaiming_init', type=bool, default=True)
    
    parsers.add_argument('--batch_size', type=int, default=48)
    parsers.add_argument('--test_batch_size', type=int, default=1)
    parsers.add_argument('--step_size', type=int, default=100)
    parsers.add_argument('--gamma', type=float, default=0.5)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=0.0001)
    parsers.add_argument('--epochs', type=int, default=300)
    
    args = parsers.parse_args()
    
    if args.dataset_name == "ChairV2":
        num_classes = 19
    else:
        num_classes = 15
        
    model = BiLSTM_SBIR(args=args)
    
    backbones_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_bacbkbone.pth")
    attention_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_attention.pth")
    linear_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_linear.pth")
    
    model.sample_embedding_network.load_state_dict(backbones_state['sample_embedding_network'], strict=False)
    model.attention.load_state_dict(attention_state['attention'], strict=False)
    model.linear.load_state_dict(linear_state['linear'])
    model.sketch_embedding_network.load_state_dict(backbones_state['sketch_embedding_network'], strict=False)
    model.sketch_attention.load_state_dict(attention_state['sketch_attention'], strict=False)
    model.sketch_linear.load_state_dict(linear_state['sketch_linear'])
        
    train_model(model, args)