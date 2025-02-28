import torch
import torch.nn.functional as F

# Các tensor đã cho
anchor = torch.tensor([
    [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
    [[3.0, 3.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
])  # shape (2, 3, 4)

positive = torch.tensor([
    [[1.1, 2.1, 3.1, 4.1]],
    [[3.1, 3.1, 3.1, 3.1]]
])  # shape (2, 1, 4)

negative = torch.tensor([
    [[4.1, 3.1, 2.1, 1.1]],
    [[1.1, 1.1, 1.1, 1.1]]
])  # shape (2, 1, 4)

margin = 0.2
batch_size = anchor.shape[0]
num_anchors = anchor.shape[1]
losses = []

# Lặp qua từng batch
for i in range(batch_size):
    batch_losses = []
    
    # Lặp qua từng anchor trong batch
    for j in range(num_anchors):
        a = anchor[i, j]  # Lấy một anchor vector (shape: 4)
        p = positive[i, 0]  # Lấy positive vector tương ứng (shape: 4)
        n = negative[i, 0]  # Lấy negative vector tương ứng (shape: 4)
        
        # Tính khoảng cách Euclidean
        dist_pos = torch.sqrt(torch.sum((a - p) ** 2))
        dist_neg = torch.sqrt(torch.sum((a - n) ** 2))
        
        # Tính triplet loss
        loss = F.relu(dist_pos - dist_neg + margin)
        batch_losses.append(loss)
    
    # Thêm loss của batch vào danh sách losses
    losses.append(torch.stack(batch_losses))

# In ra loss cho từng cặp
print("Triplet loss cho từng cặp:")
all_losses = torch.stack(losses)
print(all_losses)

# Tính loss trung bình
mean_loss = torch.mean(torch.stack(losses))
print("\nTriplet margin loss trung bình:")
print(mean_loss.item())