import os, random
from PIL import Image
import torchvision.transforms as transforms

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

def load_dataset(data_path, last_class_id, split_ratio=0.6):
    data_store = {"al":{}, "rs":{}, "test":{}}
    data_store["al"] = {"train":{"file_list":[], "labels":[]}, "valid":{"file_list":[], "labels":[]}}
    data_store["rs"] = {"train":{"file_list":[], "labels":[]}, "valid":{"file_list":[], "labels":[]}}
    data_store["test"] = {"file_list":[], "labels":[]}

    for c in range(1, (last_class_id+1)):
        train_path = os.path.join(data_path, "train", str(c))
        test_path = os.path.join(data_path, "valid", str(c))
        
        trains = [os.path.join(train_path, t) for t in os.listdir(train_path)]
        shuffled = random.sample(tmp_files_tr, len(tmp_files_tr))
            
        unit_train = shuffled[:int(len(shuffled)*split_ratio))]
        unit_val = shuffled[int(len(shuffled)*split_ratio)):]
        unit_test = [os.path.join(test_path, v) for v in os.listdir(test_path)]
            
        data_store["al"]["train"]["file_list"].extend(unit_train)
        data_store["al"]["valid"]["file_list"].extend(unit_val)
        data_store["rs"]["train"]["file_list"].extend(unit_train)
        data_store["rs"]["valid"]["file_list"].extend(unit_val)
        data_store["test"]["file_list"].extend(unit_test)

        data_store["al"]["train"]["labels"].extend([c]*len(unit_train))
        data_store["al"]["valid"]["labels"].extend([c]*len(unit_val))
        data_store["rs"]["train"]["labels"].extend([c]*len(unit_train))
        data_store["rs"]["valid"]["labels"].extend([c]*len(unit_val))
        data_store["test"]["labels"].extend([c]*len(unit_test))
            
    return data_store

def update_data_store(data_store, idx_list, sample_type="al"):

    target_files = [data_store[sample_type]["valid"]["file_list"][i] for i in idx_list]
    _target_files = [data_store[sample_type]["valid"]["file_list"][ii] for ii not in idx_list]

    target_labels = [data_store[sample_type]["valid"]["labels"][j] for j in idx_list]
    _target_labels = [data_store[sample_type]["valid"]["labels"][jj] for j not in idx_list]
    
    data_store[sample_type]["train"]["file_list"].extend(target_files)
    data_store[sample_type]["train"]["labels"].extend(target_labels)

    data_store[sample_type]["valid"]["file_list"] = _target_files
    data_store[sample_type]["valid"]["labels"] = _target_labels

    return data_store

def update_loaders(loaders, data_store, batch_size):
    al_train_dset = FlowerDataset(data_store["al"]["train"])
    al_valid_dset = FlowerDataset(data_store["al"]["valid"])
    rs_train_dset = FlowerDataset(data_store["rs"]["train"])
    rs_valid_dset = FlowerDataset(data_store["rs"]["valid"])
  
    al_trainloader = DataLoader(al_train_dset, batch_size=batch_size, shuffle=False)
    al_validloader = DataLoader(al_valid_dset, batch_size=batch_size, shuffle=False)
    rs_trainloader = DataLoader(rs_train_dset, batch_size=batch_size, shuffle=False)
    rs_validloader = DataLoader(rs_valid_dset, batch_size=batch_size, shuffle=False)
    
    loaders["al"]["train"] = al_trainloader
    loaders["al"]["valid"] = al_validloader
    loaders["rs"]["train"] = rs_trainloader
    loaders["rs"]["valid"] = rs_validloader
    
    return loaders

class FlowerDataset(Dataset):
    # Flower Dataset class    
    def __init__(self, data_store, transform=None):
        self.file_list = data_store["file_list"]
        self.labels = data_store["labels"] 
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = Image.open(self.file_list[idx])
        label = int(self.labels[idx]) -1 # for zero indexing
        
        if self.transform is None:
            self.transform = get_transform((224,224))
            image_transform = self.transform(image)
        else:
            image_transform = self.transform(image)
        
        return image_transform, label
