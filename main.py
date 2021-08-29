import os, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18

from utils import *
from config import get_params
from sampler import ActiveLearning


class ALSimulator:
    def __init__(self, data_path, arch, threshold, sampling_rate, batch_size, epoch, device, last_class_id):
        self.data_path = data_path
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.epoch = epoch 
        self.data_store = {
            "train": {"file_list":[], "labels":[]}, 
            "valid": {"file_list":[], "labels":[]}, 
            "test": {"file_list":[], "labels":[]}
            }
        self.arch = arch
        self.device = device
        self.last_class_id = last_class_id

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.iteration = 1
        self.validation_perf = {"acc":[], "loss":[]}
        self.test_perf = {"acc":[], "loss":[]}

    def load_dataset(self):

        for c in range(1, (self.last_class_id+1)):
            train_path = os.path.join(self.data_path, "train", str(c))
            valid_path = os.path.join(self.data_path, "valid", str(c))
        
            tmp_files_tr = [os.path.join(train_path, t) for t in os.listdir(train_path)]
            tmp_files_val = [os.path.join(valid_path, v) for v in os.listdir(valid_path)]
            
            self.data_store["train"]["file_list"].extend(tmp_files_tr)
            self.data_store["valid"]["file_list"].extend(tmp_files_val)
            self.data_store["train"]["labels"].extend([c]*len(tmp_files_tr))
            self.data_store["valid"]["labels"].extend([c]*len(tmp_files_val))
            
        test_path = os.path.join(self.data_path, "test")
        test_list = [os.path.join(test_path, f) for f in os.listdir(test_path)]        
        self.data_store["test"]["file_list"] = test_list
        self.data_store["test"]["labels"] = [-1]*len(test_list)

        return
    
    def setup(self, use_pretrained=True):
        
        self.load_dataset()
        train_dset = FlowerDataset(self.data_store["train"])
        valid_dset = FlowerDataset(self.data_store["valid"])
        test_dset = FlowerDataset(self.data_store["test"])

        self.train_loader = DataLoader(train_dset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dset, batch_size=self.batch_size, shuffle=False)

        if self.arch == "resnet18":
            model = resnet18(pretrained=use_pretrained, progress=True).to(self.device)
        elif self.arch == "resnet50":
            model = resnet50(pretrained=use_pretrained, progress=True).to(self.device)
        else:
            print("model not found")
        
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.last_class_id)

        return model

    def train(self, model):
        
        criterion = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.003)

        model.to(self.device)
        model.train()
        for e in range(self.epoch):
            losses = []
            for data, y in self.train_loader:
                data = data.to(self.device)
                y = y.to(self.device)

                p = model(data)
                loss = criterion(p, y)
                loss.backward()
                opt.step()
                losses.append(loss.cpu().detach().numpy().item())

            if e%5 == 0:
                print(f" -EXP iteration: {self.iteration} epoch: {e}, loss:  {sum(losses)/len(losses)}")
        
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
        p_list = []
        for data, y in self.valid_loader:
            data = data.to(self.device)
            y = y.to(self.device)
            with torch.zero_grad():
                p = model(data)
                loss = criterion(p, y)

                ps = nn.Softmax(dim=1)(p2).cpu().detach().numpy().tolist()
                p_list.extend(ps)

            tmp_embd = outputs[0][:, 0]
            embedding_outputs["embedding_output"].append(tmp_embd.cpu().detach())
            outputs = []
        
        for i, embd in enumerate(embedding_outputs["embedding_output"]):
            if i==0: 
                full_np = embd.numpy()
            else:
                full_np = np.concatenate([full_np, embd])
        
        self.iteration += 1
        return p_list, full_np

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
    
    def test(self, model):
        model.eval()
        model.to(self.device)

        y_list, p_list = [], []
        for data, _ in self.test_loader:
            data = data.to(self.device)
            y_list.extend(label.detach().numpy().tolist())
            with torch.no_grad():
                p = model(data)
                p_list.extend(p.argmax(axis=1).cpu().detach().numpy().tolist())

        return y_list, p_list

    @staticmethod
    def set_seed(random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        return

def main(als:ALSimulator, i:int):
    
    #1) baselinemodel training and testing
    baseline_model = als.baseline_trainer()
    #2) active learning with validset
    p_list, embd_feat = als.get_cnn_features(baseline_model)
    #3) additional training
    samle_idxs = als.sample_dataset(p_list, embd_feat)
    #4) testing and comparing
    al_result = 1
    
    print(f"{i}-th iteration end --- / performance AL: {al_result}")
    return


if __name__ == "__main__":
    parser = get_params()
    args = parser.parse_args()

    als = ALSimulator(
        data_path=args.data_path,
        arch=args.arch, 
        threshold=args.threshold, 
        sampling_rate=args.sampling_rate, 
        batch_size=args.batch_size, 
        epoch=args.epoch, 
        device=args.device, 
        last_class_id=5
    )

    for i in range(args.n_exp):
        als.set_seed(random.randint(1000, 9999))
        main(als, i+1)
   