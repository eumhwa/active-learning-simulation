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
            self.transform = get_standard_transform()
            image_transform = self.transform(image)
        
        return image_transform, label


def load_mnist(batch_size):
    # MNIST dataset
    mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

    return train_loader, test_loader
