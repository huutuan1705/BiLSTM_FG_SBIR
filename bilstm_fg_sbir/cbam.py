import torch
import torch.nn.functional as F

# 3 đặc trưng của phác thảo (sketch_features), mỗi vector có 2 giá trị
sketch_features = torch.tensor([[3.0, 4.0], [2.0, 3.0], [1.0, 2.0] ])

# 5 đặc trưng của ảnh (Image_Feature_ALL), mỗi vector có 2 giá trị
Image_Feature_ALL = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])

# Vị trí ảnh mục tiêu
position_query = 2

# Tính khoảng cách
all_distances = []
all_target_distances = []
for sketch_feature in sketch_features:
    distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
    target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                          Image_Feature_ALL[position_query].unsqueeze(0))
    all_distances.append(distance)
    all_target_distances.append(target_distance)

# Lấy khoảng cách nhỏ nhất
min_distance = torch.min(torch.stack(all_distances), dim=0)[0]
min_target_distance = torch.min(torch.stack(all_target_distances))

print(sketch_features.shape)
print("All distances:", all_distances)
print("Min distance:", min_distance)
print("Min target distance:", min_target_distance)