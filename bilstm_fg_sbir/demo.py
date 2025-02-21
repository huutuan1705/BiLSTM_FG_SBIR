import torch
import torch.nn.functional as F

sketch_feature = torch.randn(25, 64)
Image_Feature_ALL = torch.randn(100, 64)

distance = F.pairwise_distance(sketch_feature, Image_Feature_ALL.unsqueeze(0))

print(distance)

