import os

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18


def baseline_trainer(arch="resnet18", pretrained=True, device="cpu", data_loader):
    if arch == "resnet18":
        model = resnet18(pretrained=True, progress=True).to(device)
    elif arch == "resnet50":
        model = resnet50(pretrained=True, progress=True).to(device)
    else:
        print("model not found")
        return
    
    ## data loading 
    load_dataset()

    ## train
    model = train(model, train_loader)

    return model

def load_dataset():
    return

def train(model, train_loader):
    # hook for cnn feature extracting
    outputs = []
    def hook(module, input, output):
        outputs.append(output)

    model.layer3[-1].register_forward_hook(hook)

    # train parameters -- 

    model.train()
    for data, y in train_loader:
        data = data.to(device)
        y = y.to(device)

    
    return model

def main():
    baseline_model = baseline_trainer()
    return


if __name__ == "__main__":
    main()
   