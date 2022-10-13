import sys
sys.path.append('NeuTraLAD/')
from NeuTralAD_trainer import NeutralAD_trainer
from DataLoader import Dataset
from utils import Logger
from config.base import Grid, Config

sys.path.append('DROCC/')
from drocc import DROCCTrainer
from Net import LeNet

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
import torch.optim as optim
import argparse


def train_ntl(train_loader, test_loader=None, validation_loader=None, config_file='config/config-002.yml'):
    model_configurations = Grid(config_file, 'fmnist')
    model_config = Config(**model_configurations[0])

    model_class = model_config.model
    loss_class = model_config.loss
    optim_class = model_config.optimizer
    sched_class = model_config.scheduler
    stopper_class = model_config.early_stopper
    network = model_config.network
    trainer_class = model_config.trainer
    shuffle = model_config['shuffle'] if 'shuffle' in model_config else True

    model = model_class(network(), 1, config=model_config)
    # model.load_state_dict(torch.load(os.path.join('RESULTS/', 'model.pt')))
    optimizer = optim_class(model.parameters(),
                            lr=model_config['learning_rate'], weight_decay=model_config['l2'])

    if sched_class is not None:
        scheduler = sched_class(optimizer)
    else:
        scheduler = None


    exp_path = model_config.result_folder+model_config.exp_name  
    logger = Logger('RESULTS/experiment.log', mode='a')

    trainer = trainer_class(model, loss_function=loss_class(model_config['loss_temp']),
                     device=model_config['device']) 

    val_loss,val_auc,test_auc,test_ap,test_f1,scores,labels = \
        trainer.train(train_loader=train_loader,
                  max_epochs=model_config['training_epochs'],
                  optimizer=optimizer, scheduler=scheduler,
                  validation_loader=validation_loader, test_loader=test_loader, early_stopping=stopper_class,
                  logger=logger)
    
    return trainer


def adjust_learning_rate(epoch, total_epochs, only_ce_epochs, learning_rate, optimizer):
        """Adjust learning rate during training.
        Parameters
        ----------
        epoch: Current training epoch.
        total_epochs: Total number of epochs for training.
        only_ce_epochs: Number of epochs for initial pretraining.
        learning_rate: Initial learning rate for training.
        """
        #We dont want to consider the only ce 
        #based epochs for the lr scheduler
        epoch = epoch - only_ce_epochs
        drocc_epochs = total_epochs - only_ce_epochs
        # lr = learning_rate
        if epoch <= drocc_epochs:
            lr = learning_rate * 0.01
        if epoch <= 0.80 * drocc_epochs:
            lr = learning_rate * 0.1
        if epoch <= 0.40 * drocc_epochs:
            lr = learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer
    
    
def train_drocc(train_loader, test_loader, radius=0.2, lamda=1, lr=0.001, gamma=2.0, device='cuda', 
                optim_idx=0, mom=0, metric='AUC', epochs=70, ascent_step_size=0.001, oce=50):
    
    
    # dataset = CIFAR10_Dataset("data", 3)
    # train_loader, test_loader = dataset.loaders(batch_size=128)
    
    
    model = LeNet().to(device)
    model = nn.DataParallel(model)

    if optim_idx == 1:
        optimizer = optim.SGD(model.parameters(),
                                  lr=lr,
                                  momentum=mom)
        print("using SGD")
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=lr)
        print("using Adam")
        
    
    print(f'Raduis: {radius}')
    print(f'Lamda: {lamda}')
    trainer = DROCCTrainer(model, optimizer, lamda, radius, gamma, device)

    
    trainer.train(train_loader, test_loader, lr, adjust_learning_rate, epochs,
        metric=metric, ascent_step_size=ascent_step_size, only_ce_epochs=oce)
    
    trainer.save(model_dir)
    

    return trainer

def train(modelName='ntl', source='Datasets/datasets-01/', batch_size=16, 
          lr=0.0001, epochs=100, config_file='config/config.yml', 
          radius=0.2, lamda=1, optim_idx=0, show_roc_curve=True, show_histogram=False):
    
    if modelName in ['ntl', 'drocc']:
        train_datasets = Dataset(source=f'{source}/train', train=True, batch_size=batch_size, num_workers=0)
        test_datasets = Dataset(source=f'{source}/test', train=False, batch_size=batch_size, num_workers=0)

        dataset = (train_datasets.load(), 
                   test_datasets.load())

        train_loader, test_loader = dataset
        
    else:
        print('Model does not exist !')
        sys.exit()
    
    if modelName=='ntl':
        model = train_ntl(train_loader, test_loader=test_loader, validation_loader=test_loader, config_file=config_file)
        
    elif modelName=='drocc':
        model = train_drocc(train_loader, test_loader, radius=radius, lamda=lamda, lr=lr, device='cuda', 
                            optim_idx=optim_idx, mom=0, metric='AUC', epochs=70, ascent_step_size=0.001, oce=50)
        
    
    labels, scores, test_auc = test(model, modelName, test_loader, show_roc_curve, show_histogram)
    
    return model


def test(model, modelName, test_loader, show_roc_curve=True, show_histogram=False):
    
    
    if modelName=='ntl':
        test_auc, test_ap,test_f1, testin_loss,testout_loss, scores, labels = model.detect_outliers(test_loader, cls=None)
        
    elif modelName=='drocc':
        labels, scores, test_auc = model.test(test_loader, 'AUC')
        
    if show_roc_curve:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fig = plt.figure(figsize=(8, 4))
        # plt.title(f'Receiver Operating Characteristic - VAE - TEST ID.{id}')
        plt.plot(fpr, tpr, label=f'ROC curve [AUC: {round(test_auc, 2)}]')
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend()
        plt.show() 
        
    fontdict = {'family': 'serif',
                'color':  'black',
                'weight': 'bold',
                'size': 16,
            }

    if show_histogram:
        scores_walk = scores[np.where(categories=='Walking')]
        scores_run = scores[np.where(categories=='Anomalies')]
        scores_anomaly = scores[np.where(labels==1)]
        hist = plt.figure(figsize=(8, 4))
        plt.hist(scores_walk, alpha=1, bins='auto', label='walk')
        # plt.hist(scores_run, alpha=0.8, bins=50, label='run')
        plt.hist(scores_anomaly, alpha=0.6, bins=50, label='Anomaly')
        # plt.title(f'HISTOGRRAM - ANOMALY SCORES - TEST ID.{id}', fontdict=fontdict)
        plt.xlabel('ANOMALY SCORE', fontdict=fontdict)
        plt.legend()
        plt.show()
        
    return labels, scores, test_auc