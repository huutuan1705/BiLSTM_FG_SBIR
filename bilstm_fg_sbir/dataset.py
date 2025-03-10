import os
import torch
import random
import pickle 

from PIL import Image
from random import randint
from torch.utils.data import Dataset
from utils import get_transform
from rasterize import rasterize_sketch
import torchvision.transforms.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Dataset(Dataset):
    def __init__(self, args, mode, on_fly=False):
        super().__init__()
        self.args = args
        self.mode = mode
        self.on_fly = on_fly
        
        coordinate_path = os.path.join(args.root_dir, args.dataset_name, args.dataset_name + '_Coordinate')
        self.root_dir = os.path.join(args.root_dir, args.dataset_name)
        
        with open(coordinate_path, 'rb') as f:
            self.coordinate = pickle.load(f)
            
        self.train_sketch = [x for x in self.coordinate if 'train' in x]
        self.test_sketch = [x for x in self.coordinate if 'test' in x]
        
        self.train_transform = get_transform('train')
        self.test_transform = get_transform('test')
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_sketch)

        return len(self.test_sketch)
    
    def __getitem__(self, item):
        sample = {}
        
        if self.mode == 'train':
            sketch_path = self.train_sketch[item]
            
            positive_sample = '_'.join(self.train_sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            
            posible_list = list(range(len(self.train_sketch)))
            posible_list.remove(item)
            
            negative_item = posible_list[randint(0, len(posible_list)-1)]
            negative_sample = '_'.join(self.train_sketch[negative_item].split('/')[-1].split('_')[:-1])
            negative_path = os.path.join(self.root_dir, 'photo', negative_sample + '.png')
            
            vector_x = self.coordinate[sketch_path]
            
            list_sketch_imgs = rasterize_sketch(vector_x, self.args.num_anchors)
            if self.on_fly:
                sketch_imgs = [Image.fromarray(sk_img).convert("RGB") for sk_img in list_sketch_imgs]

            else:
                sketch_imgs = self.train_transform(Image.fromarray(list_sketch_imgs[-1]).convert("RGB"))

            positive_image = Image.open(positive_path).convert("RGB")
            negative_image = Image.open(negative_path).convert("RGB")
            
            n_flip = random.random()
            if n_flip > 0.5:

                if self.on_fly == False:
                    sketch_imgs = F.hflip(sketch_imgs)
                else:
                    sketch_imgs = [F.hflip(sk_img) for sk_img in sketch_imgs]

                positive_image = F.hflip(positive_image)
                negative_image = F.hflip(negative_image)
            
            positive_image = self.train_transform(positive_image)
            negative_image = self.train_transform(negative_image)
            
            if self.on_fly == False:
                sketch_imgs = self.train_transform(sketch_imgs)
            else:
                sketch_imgs = torch.stack([self.train_transform(sk_img) for sk_img in sketch_imgs])
                # sketch_imgs = torch.cat(sketch_imgs).view(-1, 3, 299, 299)
                
            sample = {
                'sketch_imgs': sketch_imgs, 'sketch_path': sketch_path,
                'positive_img': positive_image, 'positive_path': positive_path,
                'negative_img': negative_image, 'negative_path': negative_sample 
            }
        
        else:
            sketch_path = self.test_sketch[item]
            positive_sample = '_'.join(self.test_sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_image = self.test_transform(Image.open(positive_path).convert("RGB"))
            
            vector_x = self.coordinate[sketch_path]
            list_sketch_imgs = rasterize_sketch(vector_x, self.args.num_anchors)
            if self.on_fly:
                sketch_imgs = torch.stack([self.train_transform(Image.fromarray(sk_img).convert("RGB")) for sk_img in list_sketch_imgs])
                # print("sketch_imgs.shape: ", sketch_imgs.shape)

            else:
                sketch_imgs = self.test_transform(Image.fromarray(list_sketch_imgs[-1]).convert("RGB"))
                
            sample = {
                'sketch_imgs': sketch_imgs, 'sketch_path': sketch_path,
                'positive_img': positive_image, 'positive_path': positive_sample
            }
            
        return sample