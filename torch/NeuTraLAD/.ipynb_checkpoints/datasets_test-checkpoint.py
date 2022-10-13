import os
import pickle
import numpy as np
import logging
import random



def TestLoader(source='datasets/test/'):
    
    # extract contents
    Contents = [f'{source}/{content}' 
                for content in os.listdir(source) 
                if (os.path.isdir(f'{source}/{content}') and not content.startswith('.'))]
    
    # check is data are normal or abnormal
    IsNormal = [os.path.basename(content) in ['Walking', 'Running', 'Jogging', 'TalkOnThePhone']
                for content in Contents]
    
    datasets = []
    labels = []
    index = []
    categories = []
    for content, is_normal in zip(Contents, IsNormal):
        # label: 0 -> normal, 1 -> abnormal
        label = 0 if is_normal else 1
        AnimationType = os.path.basename(content)
        
        
        files = [os.path.join(content, f) 
                 for f in os.listdir(content)
                 if f.endswith('pickle')
                ]

        # extract short time fourier files
        sxx = []
        idx = 0
        for file in files:
            with open(file, 'rb') as f:
                sxx += pickle.load(f)

        # take the results of each receiver
        for sx in sxx:
            for i in range(sx.shape[1]):
                s = sx[:, i]
                datasets.append(s)
                labels.append(label)
                categories.append(AnimationType)
                index.append(idx+1)

    datasets = np.array(datasets)
    labels = np.array(labels)
    categories = np.array(categories)
    
    # take the magnitude   
    datasets = abs(datasets)
    
    datasets = 10 * np.log10(datasets ** 2)
    
    unique_categories = np.unique(categories, axis=0)
    for c in unique_categories:
        
        # Normalize
        MIN = np.min(datasets[np.where(categories==c)])
        MAX = np.max(datasets[np.where(categories==c)])
        datasets[np.where(categories==c)] = (datasets[np.where(categories==c)] - MIN) / (MAX - MIN)
    
    
    datasets = np.expand_dims(datasets, -1).astype('float32')
    
    return datasets, labels, np.array(categories), np.array(index)