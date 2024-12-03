import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset, ConcatDataset
import os
import pickle
import matplotlib.pyplot as plt
import argparse
import random
from pynvml import *
from utils import *


parser = argparse.ArgumentParser()

data_score_path = 'memscore'

# Data score order: from high to low (RSW H->L / SWMR H->L)
def slide_window(dataset_name, model, total_epochs, train_list, width, stride):
    data_path = f"./{data_score_path}/memscore_{dataset_name}_{model}.csv"
    s=pd.read_csv(data_path)
    s=np.array(s)

    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=True, download=False, transform=transform)
    elif dataset_name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=True, download=False, transform=transform)
    elif dataset_name == "cinic":
        with open("./data/datasets/cinic/cinic.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "texas100":
        with open("./data/datasets/texas/texas100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "purchase100":
        with open("./data/datasets/purchase100/purchase100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "location":
        with open("./data/datasets/location/location.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    
    if dataset_name == "texas100" or dataset_name == "purchase100":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    elif dataset_name == "location":
        class_indices = [[] for _ in range(30)]
        class_score_indices = [[] for _ in range(30)]
    elif dataset_name == "cinic":
        class_indices = [[] for _ in range(10)]
        class_score_indices = [[] for _ in range(10)]
    else:
        class_indices = [[] for _ in range(len(dataset.classes))]
        class_score_indices = [[] for _ in range(len(dataset.classes))]

    for idx in train_list:
        _, target = dataset[idx]
        class_indices[target].append(idx)
        class_score_indices[target].append(s[idx][1])

    sorted_list = []
    sorted_dict = []
    for one_class_idx, one_class_value in zip(class_indices, class_score_indices):
        class_dict = dict(zip(one_class_idx, one_class_value))
        sort_score = sorted(class_dict.items(), key=lambda x:x[1], reverse=True)
        sorted_dict.append(sort_score)
        temp_list = []
        for i in range(len(sort_score)):
            temp_list.append(sort_score[i][0])
        sorted_list.append(temp_list)                         # "sorted_list" includes 10 sublists (for example, CIFAR10 and CINIC), corresponding with 10 classes, and the elements    
                                                             # in per sublist is data's idx sorted by data mem-score 
    # slide window on dataset
    epochs_data_idx = []                                                       
    for i in range(total_epochs):
        data_idx = []
        start = stride * i
        end = start + width
        for classlist in sorted_list:
            if start >= len(classlist):
                continue
            if end > len(classlist):
                end = len(classlist)
            data_idx.extend(classlist[start:end])
        epochs_data_idx.append(data_idx)
    
    return epochs_data_idx

# Data score order reverse: from low to high  (RSW L->H / SWMR L->H)
def slide_window_reverse(dataset_name, model, total_epochs, train_list, width, stride):
    data_path = f"./{data_score_path}/memscore_{dataset_name}_{model}.csv"
    s=pd.read_csv(data_path)
    s=np.array(s)

    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=True, download=False, transform=transform)
    elif dataset_name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=True, download=False, transform=transform)
    elif dataset_name == "cinic":
        with open("./data/datasets/cinic/cinic.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    
    elif dataset_name == "texas100":
        with open("./data/datasets/texas/texas100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "purchase100":
        with open("./data/datasets/purchase100/purchase100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "location":
        with open("./data/datasets/location/location.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    if dataset_name == "texas100" or dataset_name == "purchase100":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    elif dataset_name == "location":
        class_indices = [[] for _ in range(30)]
        class_score_indices = [[] for _ in range(30)]
    elif dataset_name == "cinic":
        class_indices = [[] for _ in range(10)]
        class_score_indices = [[] for _ in range(10)]
    else:
        class_indices = [[] for _ in range(len(dataset.classes))]
        class_score_indices = [[] for _ in range(len(dataset.classes))]
    
    for idx in train_list:
        _, target = dataset[idx]
        class_indices[target].append(idx)
        class_score_indices[target].append(s[idx][1])

    sorted_list = []
    sorted_dict = []
    for one_class_idx, one_class_value in zip(class_indices, class_score_indices):
        class_dict = dict(zip(one_class_idx, one_class_value))
        sort_score = sorted(class_dict.items(), key=lambda x:x[1], reverse=False)
        sorted_dict.append(sort_score)
        temp_list = []
        for i in range(len(sort_score)):
            temp_list.append(sort_score[i][0])
        sorted_list.append(temp_list)                         # "sorted_list" includes 10 sublists (for example, CIFAR10 and CINIC), corresponding with 10 classes, and the elements    
                                                             # in per sublist is data's idx sorted by data mem-score 
    # slide window on dataset
    epochs_data_idx = []                                                       
    for i in range(total_epochs):
        data_idx = []
        start = stride * i
        end = start + width
        for classlist in sorted_list:
            if start >= len(classlist):
                continue
            if end > len(classlist):
                end = len(classlist)
            data_idx.extend(classlist[start:end])
        epochs_data_idx.append(data_idx)
    
    return epochs_data_idx

# Prepare data for risky memory regularization
def ml2_process(dataset_name, model, train_list, mem_thre):
    data_path = f"./{data_score_path}/memscore_{dataset_name}_{model}.csv"
    s=pd.read_csv(data_path)
    s=np.array(s)

    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=True, download=False, transform=transform)
    elif dataset_name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=True, download=False, transform=transform)
    elif dataset_name == "cinic":
        with open("./data/datasets/cinic/cinic.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "texas100":
        with open("./data/datasets/texas/texas100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "purchase100":
        with open("./data/datasets/purchase100/purchase100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "location":
        with open("./data/datasets/location/location.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset

    if dataset_name == "texas100" or dataset_name == "purchase100":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    elif dataset_name == "location":
        class_indices = [[] for _ in range(30)]
        class_score_indices = [[] for _ in range(30)]
    elif dataset_name == "cinic":
        class_indices = [[] for _ in range(10)]
        class_score_indices = [[] for _ in range(10)]
    else:
        class_indices = [[] for _ in range(len(dataset.classes))]
        class_score_indices = [[] for _ in range(len(dataset.classes))]
    
    for idx in train_list:
        _, target = dataset[idx]
        class_indices[target].append(idx)
        class_score_indices[target].append(s[idx][1])

    risk_idx = []
    gen_idx = []
    for one_class_idx, one_class_value in zip(class_indices, class_score_indices):
        for i, score in enumerate(one_class_value):
            if score >= mem_thre:
                risk_idx.append(one_class_idx[i])
            else:
                gen_idx.append(one_class_idx[i])
    return risk_idx, gen_idx

# random order
def slide_random(dataset_name, total_epochs, train_list, width, stride):
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=True, download=False, transform=transform)
    elif dataset_name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=True, download=False, transform=transform)
    elif dataset_name == "cinic":
        with open("./data/datasets/cinic/cinic.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "texas100":
        with open("./data/datasets/texas/texas100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "purchase100":
        with open("./data/datasets/purchase100/purchase100.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset
    elif dataset_name == "location":
        with open("./data/datasets/location/location.pkl", 'rb') as f:
            trainset, _ = pickle.load(f)
        dataset = trainset

    if dataset_name == "texas100" or dataset_name == "purchase100":
        class_indices = [[] for _ in range(100)]
        class_score_indices = [[] for _ in range(100)]
    elif dataset_name == "location":
        class_indices = [[] for _ in range(30)]
        class_score_indices = [[] for _ in range(30)]
    elif dataset_name == "cinic":
        class_indices = [[] for _ in range(10)]
        class_score_indices = [[] for _ in range(10)]
    else:
        class_indices = [[] for _ in range(len(dataset.classes))]
        class_score_indices = [[] for _ in range(len(dataset.classes))]
    
    for idx in train_list:
        _, target = dataset[idx]
        class_indices[target].append(idx)
 
    # Slide window on dataset: data mem-score is ordered randomly
    epochs_data_idx = []                                                       
    for i in range(total_epochs):
        data_idx = []
        start = stride * i
        end = start + width
        for classlist in class_indices:
            if start >= len(classlist):
                continue
            if end > len(classlist):
                end = len(classlist)
            data_idx.extend(classlist[start:end])
        epochs_data_idx.append(data_idx)
    return epochs_data_idx
        