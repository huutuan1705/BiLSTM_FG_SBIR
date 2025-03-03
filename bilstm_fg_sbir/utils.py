import torchvision.transforms as transforms

def get_transform(type):
    transform_list = [
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    if type == 'train':
        transform_list.extend([
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(15),
        ])
        
    return transforms.Compose(transform_list)