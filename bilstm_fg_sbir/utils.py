import torchvision.transforms as transforms

def get_transform(type):
    transform_list = []
    if type == 'train':
        transform_list.extend([
            transforms.Resize(320),
            transforms.CenterCrop(299),
        ])
    else:
        transform_list.extend([transforms.Resize(299)])
        
    transform_list.extend(
        [transforms.ToTensor(), 
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)
    