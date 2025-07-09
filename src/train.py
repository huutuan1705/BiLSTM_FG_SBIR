import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data 
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim

from src.dataset import FGSBIR_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    
    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))
    
    return dataloader_train, dataloader_test

def evaluate_model(model, dataloader_test):
    with torch.no_grad():
        model.eval()
        sketch_array_tests = []
        sketch_names = []
        image_array_tests = torch.FloatTensor().to(device)
        image_names = []
        
        for idx, batch in enumerate(tqdm(dataloader_test)):
            sketch_features_all = torch.FloatTensor().to(device)
            for data_sketch in batch['sketch_imgs']:
                sketch_feature = model.sketch_attention(model.sketch_embedding_network(data_sketch.to(device)))
                sketch_features_all = torch.cat((sketch_features_all, sketch_feature.detach()))
            
            # print("sketch_feature_ALL.shape: ", sketch_features_all.shape) # (25, 2048)           
            sketch_array_tests.append(sketch_features_all.cpu())
            sketch_names.extend(batch['sketch_path'])
            
            if batch['positive_path'][0] not in image_names:
                positive_feature = model.linear(model.attention(
                    model.sample_embedding_network(batch['positive_img'].to(device))))
                image_array_tests = torch.cat((image_array_tests, positive_feature))
                image_names.extend(batch['positive_path'])
        
        # print("sketch_array_tests[0].shape", sketch_array_tests[0].shape) #(25, 2048)
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
            
            sketch_features = model.sketch_linear(model.bilstm(sampled_batch))
            
            for i_sketch in range(sampled_batch.shape[0]):
                # print("sketch_features[i_sketch].shape: ", sketch_features[i_sketch].shape)
                sketch_feature = sketch_features[i_sketch]
                target_distance = F.pairwise_distance(sketch_feature.to(device), image_array_tests[position_query].to(device))
                distance = F.pairwise_distance(sketch_feature.unsqueeze(0).to(device), image_array_tests.to(device))
                
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
    
def train_model(model, args):
    dataloader_train, dataloader_test = get_dataloader(args)
    if args.load_pretrained:
        model.load_state_dict(torch.load(args.pretrained))
        
    loss_fn = nn.TripletMarginLoss(margin=args.margin)
    optimizer = optim.Adam([
            {'params': model.sample_embedding_network.parameters(), 'lr': args.lr},
            {'params': model.sketch_embedding_network.parameters(), 'lr': args.lr},
        ])
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    
    top1, top5, top10 = 0, 0, 0
    for i_epoch in range(args.epochs):
        print(f"Epoch: {i_epoch+1} / {args.epochs}")
        
        losses = []
        for _, batch_data in enumerate(tqdm(dataloader_train, dynamic_ncols=False)):
            model.train()
            optimizer.zero_grad()
            
            sketch_feature, positive_feature, negative_feature = model(batch_data)
            loss = loss_fn(sketch_feature, positive_feature, negative_feature)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            
            losses.append(loss.item())
        
        avg_loss = sum(losses) / len(losses)
        top1_eval, top5_eval, top10_eval, meanA, meanB = evaluate_model(model, dataloader_test)
        
        if top5_eval > top5:
            top1, top5, top10 = top1_eval, top5_eval, top10_eval
            torch.save(model.state_dict(), "best_model.pth")
            torch.save(
                {
                    'sample_embedding_network': model.sample_embedding_network.state_dict(),
                    'sketch_embedding_network': model.sketch_embedding_network.state_dict(),
                }, args.dataset_name + '_backbone.pth')
            
            torch.save({'attention': model.attention.state_dict(),
                        'sketch_attention': model.sketch_attention.state_dict(),
                        }, args.dataset_name + '_attention.pth')
            torch.save({'linear': model.linear.state_dict(),
                        'sketch_linear': model.sketch_linear.state_dict(),
                        }, args.dataset_name + '_linear.pth')
        print('Top 1 accuracy:  {:.4f}'.format(top1_eval))
        print('Top 5 accuracy:  {:.4f}'.format(top5_eval))
        print('Top 10 accuracy: {:.4f}'.format(top10_eval))
        print('Mean A         : {:.4f}'.format(meanA))
        print('Mean B         : {:.4f}'.format(meanB))
        print('Loss:            {:.4f}'.format(avg_loss))