import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from datasets_train import TrainLoader
from datasets_test import TestLoader


class CustomDataset(Dataset):
    def __init__(self, source, train=True, transform=None, target_transform=None):
        self.source = source
        self.transform = transform
        self.target_transform = target_transform
        

        if train:
            self.stft, self.labels, self.categories, _ = TrainLoader(self.source, conjugate=True)
        else:
            self.stft, self.labels, self.categories, _ = TestLoader(self.source)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        sxx = self.stft[idx]
        label = self.labels[idx]
        category = self.categories[idx]
        
        if self.transform:
            sxx = self.transform(sxx)
        if self.target_transform:
            label = self.target_transform(label)
            
        return sxx, label, category, idx
    
    
class Dataset():
    
    def __init__(self, source, train=True, batch_size=32, num_workers=0, shuffle=False):
        self.source = source
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train = train
        
    def load(self):
            
        transform = transforms.Compose([
            ToTensor(),
    ])

        data = CustomDataset(source=self.source, train=self.train, transform=transform)
        
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)
        
        return loader