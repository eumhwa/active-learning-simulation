import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18

from utils import *
from sampler import ActiveLearning

# Global variable (config)
THRESHOLD = 0.5
SAMPLING_RATE = 0.05

# Functions
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
    
    # train parameters -- 

    model.train()
    for data, y in train_loader:
        data = data.to(device)
        y = y.to(device)

    
    return model

def get_cnn_features(model, valid_loader):
    # hook for cnn feature extracting
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    
    model.layer3[-1].register_forward_hook(hook)
    embedding_outputs = {"embedding_output":[]}

    model.eval()
    for data, _ in valid_loader:
        data = data.to(device)
        with torch.zero_grad():
            _ = model(data)

        # hook
        tmp_embd = outputs[0][:, 0]
        embedding_outputs["embedding_output"].append(tmp_embd.cpu().detach())
        outputs = []
    
    for i, embd in enumerate(embedding_outputs["embedding_output"]):
        if i==0: 
            full_np = embd.numpy()
        else:
            full_np = np.concatenate([full_np, embd])
    
    return full_np

def sample_dataset(prob_list, embd_features, valid_dataset):
    al = ActiveLearning(
        prob_list,
        embd_features,
        THRESHOLD,
        SAMPLING_RATE
    )
    etp_pts = al.entropy_sampling()
    k_centers = al.core_set_selection()
    al_sample_idx = list(set(etp_pts + k_centers))
    
    return [valid_dataset[i] for i in al_sample_idx]


def main(i):
    #1) train/val/test split
    #2) baselinemodel training and testing
    baseline_model = baseline_trainer()
    #3) active learning with validset
    #4) additional training
    #5) testing and comparing
    al_result = 1
    rs_result = 1
    print(f"{i}-th iteration end --- / performance AL: {al_result} and RS: {rs_result}")
    return


if __name__ == "__main__":
    N = 50
    for i in range(50):
        main(i)
   