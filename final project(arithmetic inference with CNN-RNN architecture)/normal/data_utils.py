import os
from re import L
import torch
import numpy as np

from torch.utils.data import Dataset


class Mydataset(Dataset):
    def __init__(self, data_path, transform=None, train=True):
        self.path = data_path
        self.transform = transform
        if train:
            len_data = 50000
            len_slices = len_data // 10
            self.data = []
            self.label = []
            for i in range(10):
                start_idx, end_idx = i * len_slices, (i+1) * len_slices
                print(i, 'load from %d to %d' % (start_idx, end_idx))
                cur_load_path = os.path.join(data_path, 'train_data_' + str(i) + '.npy')                
                cur_load_dataset = np.load(cur_load_path, allow_pickle=True).item()
                self.data.extend(cur_load_dataset['train_data'])
                self.label.extend(cur_load_dataset['train_label'])
        else:
            len_data = 10000
            len_slices = len_data // 2
            self.data = []
            self.label = []
            for i in range(2):
                start_idx, end_idx = i * len_slices, (i+1) * len_slices
                print(i, 'load from %d to %d' % (start_idx, end_idx))
                cur_load_path = os.path.join(data_path, 'valid_data_' + str(i) + '.npy')
                cur_load_dataset = np.load(cur_load_path, allow_pickle=True).item()
                self.data.extend(cur_load_dataset['valid_data'])
                self.label.extend(cur_load_dataset['valid_label'])

    def __getitem__(self, idx):
        data, label = self.data[idx], self.label[idx]        
        data_out = torch.zeros(data.shape)
        label_out = torch.tensor(np.array(label, dtype=int))
        if self.transform is not None:
            for i in range(data.shape[0]):
                data_seg = data[i, :, :]  # (H, W)
                data_transform = self.transform(data_seg.astype(np.uint8))  # (H, W) -> (1, H, W)
                data_out[i, :, :] = data_transform.squeeze(0)  # (1, H, W) -> (H, W)
        else:
            data_out = torch.tensor(data)
        return data_out, label_out

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    img = [item[0].unsqueeze(1) for item in batch]
    label = [item[1] for item in batch]
    
    return img, label