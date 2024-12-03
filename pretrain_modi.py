"""
This code is modified from https://github.com/Machine-Learning-Security-Lab/mia_prune
"""
import argparse
import json
import os
import pickle
import random
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from base_model import BaseModel
from datasets import get_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Subset
from utils import seed_worker


parser = argparse.ArgumentParser()
parser.add_argument('device', default=0, type=int, help="GPU id to use")
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--dataset_name', default='cifar10', type=str)
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--image_size', default=32, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--early_stop', default=5, type=int, help="patience for early stopping")
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--optimizer', default="adam", type=str)
parser.add_argument('--shadow_num', default=5, type=int)


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    save_folder = f"./result/{args.dataset_name}_{args.model_name}"

    print(f"Save Folder: {save_folder}")
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    
    if testset is None:
        total_dataset = trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
    total_size = len(total_dataset)
    data_path = f"{save_folder}/data_prepare.pkl"

    # split the dataset into victim dataset and shadow dataset, then split each into train, val, test
    # In our setting, we should ensure the train-set completely from "trining set", the "training set" is default training data (e.g., downloaded cifar10/100...)
    # or artificial split training data (eg. purchase/texas), we need to use mem-score in next process
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    victim_list, attack_list = train_test_split(list(range(len(trainset))), test_size=0.5, random_state=args.seed)    
    #victim_train_list, temp_test_list = train_test_split(victim_list, test_size=0.2, random_state=args.seed)    #texas100/purchase100 test_size=0.2/0.1
    victim_train_list, temp_test_list = train_test_split(victim_list, test_size=0.1, random_state=args.seed)
    temp_test_dataset1 = Subset(trainset, temp_test_list)
    test_data1 = ConcatDataset([temp_test_dataset1, testset])
    victim_dev_list, victim_test_list = train_test_split(list(range(len(test_data1))), test_size=0.66, random_state=args.seed)
    attack_split_list = []
    shadow_train_list = []
    for i in range(args.shadow_num):
        #attack_train_list, temp_test_list = train_test_split(attack_list, test_size=0.2, random_state=args.seed + i)
        attack_train_list, temp_test_list = train_test_split(attack_list, test_size=0.1, random_state=args.seed + i)
        temp_test_dataset2 = Subset(trainset, temp_test_list)
        test_data2 = ConcatDataset([temp_test_dataset2, testset])
        attack_dev_list, attack_test_list = train_test_split(list(range(len(test_data2))), test_size=0.66, random_state=args.seed + i)
        attack_train_dataset = Subset(trainset, attack_train_list)
        attack_dev_dataset = Subset(test_data2, attack_dev_list)
        attack_test_dataset = Subset(test_data2, attack_test_list)
        shadow_train_list.append(attack_train_list)
        attack_split_list.append([attack_train_dataset, attack_dev_dataset, attack_test_dataset])

    # Train the victim model
    victim_train_dataset = Subset(trainset, victim_train_list)
    victim_dev_dataset = Subset(test_data1, victim_dev_list)
    victim_test_dataset = Subset(test_data1, victim_test_list)

    with open(data_path, 'wb') as f:
        pickle.dump([victim_train_list, victim_train_dataset, victim_dev_dataset, victim_test_dataset, attack_split_list, shadow_train_list], f)

    print(f"Total Data Size: {total_size}, "
          f"Victim Train Size: {len(victim_train_list)}, "
          f"Victim Dev Size: {len(victim_dev_list)}, "
          f"Victim Test Size: {len(victim_test_list)}")

    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     pin_memory=True, worker_init_fn=seed_worker)
    victim_dev_loader = DataLoader(victim_dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                   pin_memory=True, worker_init_fn=seed_worker)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True, worker_init_fn=seed_worker)

    victim_model_save_folder = save_folder + "/victim_model"

    print("Train Victim Model")
    if not os.path.exists(victim_model_save_folder):
        os.makedirs(victim_model_save_folder)
    victim_model = BaseModel(
        args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=victim_model_save_folder,
        device=device, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
    best_acc = 0
    count = 0
    for epoch in range(args.epochs):
        train_acc, train_loss = victim_model.train(victim_train_loader, f"Epoch {epoch} Train")
        dev_acc, dev_loss = victim_model.test(victim_dev_loader, f"Epoch {epoch} Dev")
        test_acc, test_loss = victim_model.test(victim_test_loader, f"Epoch {epoch} Test")
        if dev_acc > best_acc:
            best_acc = dev_acc
            save_path = victim_model.save(epoch, test_acc, test_loss)
            best_path = save_path
            count = 0
        elif args.early_stop > 0:
            count += 1
            if count > args.early_stop:
                print(f"Early Stop at Epoch {epoch}")
                break
    shutil.copyfile(best_path, f"{victim_model_save_folder}/best.pth")

    # Train shadow models
    for shadow_ind in range(args.shadow_num):
        attack_train_dataset, attack_dev_dataset, attack_test_dataset = attack_split_list[shadow_ind]
        attack_train_loader = DataLoader(attack_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                         pin_memory=True, worker_init_fn=seed_worker)
        attack_dev_loader = DataLoader(attack_dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                       pin_memory=True, worker_init_fn=seed_worker)
        attack_test_loader = DataLoader(attack_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                        pin_memory=True, worker_init_fn=seed_worker)

        print(f"Train Shadow Model {shadow_ind}")
        shadow_model_save_folder = f"{save_folder}/shadow_model_{shadow_ind}"
        if not os.path.exists(shadow_model_save_folder):
            os.makedirs(shadow_model_save_folder)
        shadow_model = BaseModel(
            args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=shadow_model_save_folder,
            device=device, optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
        best_acc = 0
        count = 0
        for epoch in range(args.epochs):
            train_acc, train_loss = shadow_model.train(attack_train_loader, f"Epoch {epoch} Shadow Train")
            dev_acc, dev_loss = shadow_model.test(attack_dev_loader, f"Epoch {epoch} Shadow Dev")
            test_acc, test_loss = shadow_model.test(attack_test_loader, f"Epoch {epoch} Shadow Test")
            if dev_acc > best_acc:
                best_acc = dev_acc
                save_path = shadow_model.save(epoch, test_acc, test_loss)
                best_path = save_path
                count = 0
            elif args.early_stop > 0:
                count += 1
                if count > args.early_stop:
                    print(f"Early Stop at Epoch {epoch}")
                    break

        shutil.copyfile(best_path, f"{shadow_model_save_folder}/best.pth")


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    print(args)
    main(args)
