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
BATCH_SIZE = 4
DATA_PATH = ""

# Functions
class ALSimulator:
    def __init__(self, arch, threshold, sampling_rate, batch_size, epoch, device, last_class_id):
        self.data_path = "./data/flower_data"
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.epoch = epoch
        fl = {"file_list":[], "labels":[]}
        self.data_store = {"train": fl, "valid": fl, "test": fl}
        self.arch = arch
        self.device = device
        self.last_class_id = last_class_id

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def load_dataset(self):
        
        for c in range(1, (self.last_class_id+1)):
            tmp_files_tr = os.listdir(os.path.join(self.data_path, "train", str(c)))
            tmp_files_val = os.listdir(os.path.join(self.data_path, "valid", str(c)))
            
            self.data_store["train"]["file_list"].extend(tmp_files_tr)
            self.data_store["valid"]["file_list"].extend(tmp_files_val)
            self.data_store["train"]["labels"].extend([c]*len(tmp_files_tr))
            self.data_store["valid"]["labels"].extend([c]*len(tmp_files_val))
            
        test_list = os.listidr(os.path.join(self.data_path, "test"))
        self.data_store["train"]["file_list"].extend(test_list)

        return
    
    def setup(self, use_pretrained=True):
        
        self.load_dataset()
        train_dset = CustomDataset(self.data_store["train"]["file_list"], self.data_store["train"]["labels"])
        valid_dset = CustomDataset(self.data_store["valid"]["file_list"], self.data_store["valid"]["labels"])
        test_dset = CustomDataset(self.data_store["test"]["file_list"], [-1]*len(self.data_store["test"]["file_list"]))

        self.train_loader = DataLoader(train_dset, BATCH_SIZE=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dset, BATCH_SIZE=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dset, BATCH_SIZE=self.batch_size, shuffle=False)

        if self.arch == "resnet18":
            model = resnet18(pretrained=use_pretrained, progress=True).to(self.device)
        elif self.arch == "resnet50":
            model = resnet50(pretrained=use_pretrained, progress=True).to(self.device)
        else:
            print("model not found")

        return model

    def train(self, model):
        
        criterion = torch.nn.CrossEntropyLoss()
        opt = optim.SGD(model.parameters(), lr=0.01)

        model.to(self.device)
        model.train()
        for e in range(self.epoch):
            for data, y in self.train_loader:
                data = data.to(self.device)
                y = y.to(self.device)

                p = model(data)
                loss = criterion(p, y)
                loss.backward()
                opt.step()

        return model

    def baseline_trainer(self):    
        model = self.setup()
        return self.train(model)

    def update_valid_dset(self, sample_idx):
        self.data_store ## update 
        return 

    def get_cnn_features(self, model):
        # hook for cnn feature extracting
        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        
        model.layer3[-1].register_forward_hook(hook)
        embedding_outputs = {"embedding_output":[]}

        model.to(self.device)
        model.eval()
        for data, _ in self.valid_loader:
            data = data.to(self.device)
            with torch.zero_grad():
                _ = model(data)

            tmp_embd = outputs[0][:, 0]
            embedding_outputs["embedding_output"].append(tmp_embd.cpu().detach())
            outputs = []
        
        for i, embd in enumerate(embedding_outputs["embedding_output"]):
            if i==0: 
                full_np = embd.numpy()
            else:
                full_np = np.concatenate([full_np, embd])
        
        return full_np

    def sample_dataset(self, prob_list, embd_features):
        al = ActiveLearning(
            prob_list,
            embd_features,
            self.threshold,
            self.sampling_rate
        )
        etp_pts = al.entropy_sampling()
        k_centers = al.core_set_selection()
        al_sample_idx = list(set(etp_pts + k_centers))
        
        return al_sample_idx
    def validation(self, model):
        
        return 
    
    def test(self, model):
        model.eval()
        model.to(self.device)
        for data, _ in self.test_loader:
            data = data.to(self.device)
            with torch.zero_grad():
                y = model(data)

        return


def main(i):
    als = ALSimulator(
        arch="resnet18", threshold=0.5, sampling_rate=0.05, 
        batch_size=4, epoch=25, device="cpu", last_class_id=10
    )
    #1) train/val/test split
    #2) baselinemodel training and testing
    baseline_model = als.baseline_trainer()
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
   