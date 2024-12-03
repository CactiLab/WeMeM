import os
import pickle
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, Subset



def get_dataset(name, train=True):
    print(f"Build Dataset {name}")
    if name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=train, download=True, transform=transform)
    elif name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=train, download=True, transform=transform)
    elif name == "mnist":
        mean = (0.1307,)
        std = (0.3081,)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                        ])
        dataset = torchvision.datasets.MNIST(root='data/datasets/mnist-data', train=train, download=True,
                                             transform=transform)
    elif name == "cinic":
        # the dataset can be downloaded from https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=4&isAllowed=y
        if not os.path.exists("./data/datasets/cinic/cinic.pkl"):
            cinic_directory = './data/datasets/cinic'
            cinic_mean = [0.47889522, 0.47227842, 0.43047404]
            cinic_std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=cinic_mean,std=cinic_std),
            ])
            trainset = torchvision.datasets.ImageFolder(cinic_directory + '/train', transform=transform)
            testset = torchvision.datasets.ImageFolder(cinic_directory + '/test', transform=transform)

            with open("./data/datasets/cinic/train_test_idx.pkl", 'rb') as f:
                trainidx, testidx= pickle.load(f)
            train_data = Subset(trainset, trainidx)
            test_data = Subset(testset, testidx)
            with open("./data/datasets/cinic/cinic.pkl", "wb") as f:
                pickle.dump([train_data, test_data], f)
        else:
            with open("./data/datasets/cinic/cinic.pkl", "rb") as f:
                train_data, test_data = pickle.load(f)
        if train == False:
            dataset = test_data
        else:
            dataset = train_data
    
    elif name == "texas100":
        # the dataset can be downloaded from https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz
        if not os.path.exists("./data/datasets/texas/texas100.pkl"):
            x = np.loadtxt("./data/datasets/texas/feats.txt", delimiter=',')
            x_data = torch.tensor(x[:, :]).float()
            y = np.loadtxt("./data/datasets/texas/labels.txt", delimiter=',')
            y_data = torch.tensor(y[:] - 1).long()
            dataset = TensorDataset(x_data, y_data)
            trainset, testset = train_test_split(list(range(len(dataset))), test_size=0.2) # Make sure to calculate the mem-score for these training data.
            train_dataset = Subset(dataset, trainset)
            test_dataset = Subset(dataset, testset)
            with open("./data/datasets/texas/texas100.pkl", 'wb') as f:
                pickle.dump([train_dataset, test_dataset], f)
        else:
            with open("./data/datasets/texas/texas100.pkl", 'rb') as f:
                train_dataset, test_dataset = pickle.load(f)
        if train == False:
            dataset = test_dataset
        else:
            dataset = train_dataset

    elif name == "location":
        # the dataset can be downloaded from https://github.com/jjy1994/MemGuard/tree/master/data/location
        if not os.path.exists("./data/datasets/location/location.pkl"):
            dataset = np.load("./data/datasets/location/data_complete.npz")
            x_data = torch.tensor(dataset['x'][:, :]).float()
            y_data = torch.tensor(dataset['y'][:] - 1).long()
            dataset = TensorDataset(x_data, y_data)
            trainset, testset = train_test_split(list(range(len(dataset))), test_size=0.2) # Make sure to calculate the mem-score for these training data.
            train_dataset = Subset(dataset, trainset)
            test_dataset = Subset(dataset, testset)
            with open("./data/datasets/location/location.pkl", 'wb') as f:
                pickle.dump([train_dataset, test_dataset], f)
        else:
            with open("./data/datasets/location/location.pkl", 'rb') as f:
                train_dataset, test_dataset = pickle.load(f)
        if train == False:
            dataset = test_dataset
        else:
            dataset = train_dataset

    elif name == "purchase100":
        # the dataset can be downloaded from https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz
        if not os.path.exists("./data/datasets/purchase100/purchase100.pkl"):
            dataset = np.loadtxt("./data/datasets/purchase100/purchase100.txt", delimiter=',')
            x_data = torch.tensor(dataset[:, 1:], dtype=torch.float32)
            y_data = torch.tensor(dataset[:, 0] - 1, dtype=torch.long)
            dataset = TensorDataset(x_data, y_data)
            trainset, testset = train_test_split(list(range(len(dataset))), test_size=0.2) # Make sure to calculate the mem-score for these training data.
            train_dataset = Subset(dataset, trainset)
            test_dataset = Subset(dataset, testset)
            with open("./data/datasets/purchase100/purchase100.pkl", 'wb') as f:
                pickle.dump([train_dataset, test_dataset], f)
        else:
            with open("./data/datasets/purchase100/purchase100.pkl", 'rb') as f:
                train_dataset, test_dataset = pickle.load(f)
        if train == False:
            dataset = test_dataset
        else:
            dataset = train_dataset
    else:
        raise ValueError

    return dataset

