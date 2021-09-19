import os, random, json, copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18, wide_resnet50_2

from utils import *
from config import get_params
from sampler import ActiveLearning


class ALSimulator:
    def __init__(self, data_path, arch, threshold, sampling_rate, batch_size, 
                epoch, retrain_epoch, img_size, device, last_class_id):
        self.data_path = data_path
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.epoch = epoch 
        self.retrain_epoch = epoch
        self.img_shape = (img_size, img_size)
        self.arch = arch
        self.device = device
        self.last_class_id = last_class_id

        self.data_store = {}
        self.loaders = {"al":{"train":None, "valid":None}, "rs":{"train":None, "valid":None}}
        self.test_loader = None

        self.iteration = 1
    
    def setup(self, use_pretrained=True):
        
        self.data_store = load_dataset(self.data_path, self.last_class_id, split_ratio=0.6)
        train_dset = FlowerDataset(self.data_store["rs"]["train"], get_train_transform(self.img_shape))
        valid_dset = FlowerDataset(self.data_store["rs"]["valid"], get_train_transform(self.img_shape))
        test_dset = FlowerDataset(self.data_store["test"], get_test_transform(self.img_shape))

        base_trainloader = DataLoader(train_dset, batch_size=self.batch_size, shuffle=True)
        base_validloader = DataLoader(valid_dset, batch_size=self.batch_size, shuffle=False)
        self.loaders["al"]["train"] = base_trainloader
        self.loaders["rs"]["train"] = base_trainloader
        self.loaders["al"]["valid"] = base_validloader
        self.loaders["rs"]["valid"] = base_validloader
        self.test_loader = DataLoader(test_dset, batch_size=self.batch_size, shuffle=False)

        if self.arch == "resnet18":
            model = resnet18(pretrained=use_pretrained, progress=True).to(self.device)
        elif self.arch == "resnet50":
            model = resnet50(pretrained=use_pretrained, progress=True).to(self.device)
        elif self.arch == "wide_resnet50_2":
            model = wide_resnet50_2(pretrained=use_pretrained, progress=True).to(self.device)
        else:
            print("model not found")
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.last_class_id)

        return model

    def train(self, model, loader):
        
        criterion = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)

        epch = self.epoch if self.iteration==1 else self.retrain_epoch
        valid_ys = [l-1 for l in loader["valid"].dataset.labels]
        model.to(self.device)
        
        for e in range(epch):
            train_losses, train_ps, train_ys = [], [], []
            model.train()
            tqdm_loader = tqdm(
                loader["train"],
                desc="Training (X / X Epochs) (acc=X% ,loss=X.X) // Validation (acc=X%, loss=X.X)",
                bar_format="{l_bar}{r_bar}",
                dynamic_ncols=True
            )
            for data, y in tqdm_loader:
                data = data.to(self.device)
                y = y.to(self.device)
                opt.zero_grad()

                p = model(data)
                loss = criterion(p, y)
                train_losses.append(loss.cpu().detach().numpy().item())
                train_ps.extend(p.argmax(axis=1).cpu().detach().numpy().tolist())
                train_ys.extend(y.cpu().detach().numpy().tolist())
                
                loss.backward()
                opt.step()
                
            valid_losses, valid_ps = [], []
            with torch.no_grad():
                model.eval()
                for data_, y_ in loader["valid"]:
                    data_ = data_.to(self.device)
                    y_ = y_.to(self.device)

                    p_ = model(data_)
                    loss_ = criterion(p_, y_)
                    valid_losses.append(loss_.cpu().detach().numpy().item())
                    valid_ps.extend(p_.argmax(axis=1).cpu().detach().numpy().tolist())

            if e%5 == 4:
                tqdm_loader.set_description(f"Training ({e} / {epch} Epochs) (acc={str(self.performance(train_ps, train_ys))}% ,loss={sum(train_losses)/len(train_losses)}) \
                // Validation (acc={str(self.performance(valid_ps, valid_ys))}%, loss={sum(valid_losses)/len(valid_losses)})")
                #print(f"#EXP iteration: {self.iteration} epoch: {e}/{epch}")
                #print(f"##trian (acc, loss): ({str(self.performance(train_ps, train_ys))}%, {sum(train_losses)/len(train_losses)}) \
                #and valid (acc, loss): ({str(self.performance(valid_ps, valid_ys))}%, {sum(valid_losses)/len(valid_losses)})")
        
        return model

    def baseline_trainer(self):    
        model = self.setup()
        return self.train(model, self.loaders["rs"])

    def get_cnn_features(self, model):
        # hook for cnn feature extracting
        outputs = []
        def hook(module, input, output):
            outputs.append(output)
        
        model.avgpool.register_forward_hook(hook)
        embedding_outputs = {"embedding_output":[]}

        model.to(self.device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        p_list = []
        for data, y in self.loaders["al"]["valid"]:
            data = data.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                p = model(data)
                loss = criterion(p, y)

                ps = nn.Softmax(dim=1)(p).cpu().detach().numpy().tolist()
                p_list.extend(ps)

            tmp_embd = outputs[0].squeeze()
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
        random_sample_idx = al.random_sampling(len(prob_list), len(al_sample_idx))
        
        return al_sample_idx, random_sample_idx
    
    def update_valid_dset(self, al_idx, rs_idx):
        tmp_data_store = update_data_store(self.data_store, al_idx, sample_type="al")
        self.data_store = update_data_store(tmp_data_store, rs_idx, sample_type="rs")
        self.loaders = update_loaders(self.loaders, self.data_store, self.batch_size, self.img_shape)
        return 

    def test(self, model):
        model.eval()
        model.to(self.device)

        y_list, p_list = [], []
        for data, y in self.test_loader:
            data = data.to(self.device)
            y_list.extend(y.detach().numpy().tolist())
            with torch.no_grad():
                p = model(data)
                p_list.extend(p.argmax(axis=1).cpu().detach().numpy().tolist())

        return y_list, p_list

    def copy_baseline_model_wts(self, model):
        base_model_wts = copy.deepcopy(model.state_dict())
        if self.arch == "resnet18":
            new_model = resnet18(pretrained=use_pretrained, progress=True).to(self.device)
        elif self.arch == "resnet50":
            new_model = resnet50(pretrained=use_pretrained, progress=True).to(self.device)
        elif self.arch == "wide_resnet50_2":
            new_model = wide_resnet50_2(pretrained=use_pretrained, progress=True).to(self.device)
        else:
            print("model not found")
        
        num_ftrs = new_model.fc.in_features
        new_model.fc = nn.Linear(num_ftrs, self.last_class_id)
        new_model.load_state_dict(best_model_wts)
        return new_model

    @staticmethod
    def performance(ys, ps):
        oneh = []
        for y, p in zip(ys, ps):
            if y==p:
                oneh.append(1)
            else:
                oneh.append(0)
                
        return 100*(sum(oneh)/len(oneh))

    @staticmethod
    def set_seed(random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        return


def save_result():
    
    return

def main(als:ALSimulator, n_step:int, i:int):
    
    #1) baselinemodel training and testing
    
    baseline_model = als.baseline_trainer()
    base_y, base_p = als.test(baseline_model)
    perf = als.performance(base_y, base_p)
    print(f"1-th iteration in {i}-th exp end - / baseline performance: {perf}")
    
    #2) active learning with validset
    p_list, embd_feat = als.get_cnn_features(baseline_model)
    
    #3) additional training with updated train/valid set
    al_idx, rs_idx = als.sample_dataset(p_list, embd_feat)
    als.update_valid_dset(al_idx, rs_idx)

    al_model = als.copy_baseline_model_wts(baseline_model)
    rs_model = als.copy_baseline_model_wts(baseline_model)
    al_model = als.train(al_model, als.loaders["al"])
    rs_model = als.train(rs_model, als.loaders["rs"])
    
    #5) testing and comparing
    al_y, al_p = als.test(al_model)
    rs_y, rs_p = als.test(rs_model)
    al_acc = als.performance(al_y, al_p)
    rs_acc = als.performance(rs_y, rs_p)
    print(f"{n_step}-th iteration in {i}-th exp end - / performance al: {al_acc} and rs: {rs_acc}")
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
        retrain_epoch=args.re_epoch,
        img_size=args.img_size,
        device=args.device, 
        last_class_id=args.last_class_id
    )

    for i in range(args.n_exp):
        als.set_seed(random.randint(1000, 9999))
        main(als, args.n_iter, i+1)
   