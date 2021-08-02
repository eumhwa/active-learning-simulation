import os
from PIL import Image
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

def get_transform(img_size):
    trans = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return trans

class CustomDataset(Dataset):
    # Custom Dataset class    
    def __init(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = Image.open(self.file_list[idx])
        label = self.labels[idx]
        
        if self.transform is None:
            self.transform = get_transform((224,224))
            image_transform = self.transform(image)
        
        return image_transform, label
