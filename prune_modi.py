"""
This code is modified from https://github.com/Machine-Learning-Security-Lab/mia_prune
"""
import argparse
import copy
import json
import os
import time
import pickle
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from base_model import BaseModel
from datasets import get_dataset
from pruner import get_pruner
from pyvacy import analysis, optim
from data_process import *
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
parser.add_argument('--prune_epochs', default=21, type=int)
parser.add_argument('--pruner_name', default='l1unstructure', type=str)
parser.add_argument('--prune_sparsity', default=0.6, type=float)
parser.add_argument('--defend', default="", type=str, help="'' if no defense")
parser.add_argument('--adaptive', action='store_true')
parser.add_argument('--shadow_num', default=5, type=int)
parser.add_argument('--defend_arg', default=4, type=float)
parser.add_argument('--prune_iter', default=5, type=int)
parser.add_argument('--stride', type=int, help="stride of slide window")
parser.add_argument('--width', type=int, help="width of slide window")
parser.add_argument('--weight_decay_mem', default=5e-4, type=float)
parser.add_argument('--mem_thre', default=0.6, type=float)


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    prune_lr = args.lr
    if args.defend == "":
        prune_prefix = f"{args.pruner_name}_{args.prune_sparsity}"
    elif args.defend == "slide" or args.defend == "slide_re" or args.defend == "ml2" or args.defend == "slide_ml2" \
        or args.defend == "slide_re_ml2" or args.defend == "random":
        prune_prefix = f"{args.pruner_name}_{args.prune_sparsity}_{args.defend}"
    else:
        prune_prefix = f"{args.pruner_name}_{args.prune_sparsity}_{args.defend}_{args.defend_arg}"
    save_folder = f"result/{args.dataset_name}_{args.model_name}"

    os.makedirs(f'log/{args.dataset_name}_{args.model_name}', exist_ok=True)
    if args.defend == "slide" or args.defend == "slide_re" or args.defend == "slide_ml2" or args.defend == "slide_re_ml2" or args.defend == "random":
        content = f'{args.dataset_name}_{args.model_name}'+ '_' + prune_prefix + f"{'_' + str(args.stride) + '-' + str(args.width) if args.defend else ''}"
        name = f'{args.pruner_name}_{args.prune_sparsity}_{args.defend}'
    else:
        content = f'{args.dataset_name}_{args.model_name}'+ '_' + prune_prefix
        name = f'{args.pruner_name}_{args.prune_sparsity}_{args.defend}'
    with open(f'log/{args.dataset_name}_{args.model_name}/{name}.txt', 'a') as appender:
        appender.write(content + '_' + str(args.weight_decay_mem) + "\n")

    print(f"Save Folder: {save_folder}")
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    if testset is None:
        total_dataset = trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
    total_size = len(total_dataset)
    data_path = f"{save_folder}/data_prepare.pkl"

    # load data split for the pretrained victim and shadow model
    with open(data_path, 'rb') as f:
        victim_train_list, victim_train_dataset, victim_dev_dataset, victim_test_dataset, attack_split_list, shadow_train_list= pickle.load(f)

    print(f"Total Data Size: {total_size}, "
          f"Victim Train Size: {len(victim_train_dataset)}, "
          f"Victim Dev Size: {len(victim_dev_dataset)}, "
          f"Victim Test Size: {len(victim_test_dataset)}")
    

    if args.defend == "adv":
        victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                     pin_memory=True, worker_init_fn=seed_worker)
    else:
        victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     pin_memory=True, worker_init_fn=seed_worker)
    victim_dev_loader = DataLoader(victim_dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                   pin_memory=True, worker_init_fn=seed_worker)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True, worker_init_fn=seed_worker)

    victim_model_save_folder = save_folder + "/victim_model"
    # load pretrained model
    victim_model_path = f"{victim_model_save_folder}/best.pth"
    victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    victim_model.load(victim_model_path)
    test_acc, test_loss = victim_model.test(victim_test_loader, "Pretrained Victim")
    with open(f'log/{args.dataset_name}_{args.model_name}/{name}.txt', 'a') as appender:
        appender.write("Pretrained Victim: Accuracy {:.3f}, Loss {:.3f}".format(test_acc, test_loss)+ "\n")
    victim_acc = test_acc
    
    print("Prune Victim Model")
    pruned_model_save_folder = f"{save_folder}/{prune_prefix}_model"
    victim_model_path = f"{victim_model_save_folder}/best.pth"
    victim_model.load(victim_model_path)

    org_state = copy.deepcopy(victim_model.model.state_dict())
    if not os.path.exists(pruned_model_save_folder):
        os.makedirs(pruned_model_save_folder)
    
    # prune victim model
    if args.defend == "adv":
        attack_model_type = "mia_fc"
    else:
        attack_model_type = ""
    
    victim_pruned_model = BaseModel(
        args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, lr=prune_lr,
        weight_decay=args.weight_decay, save_folder=pruned_model_save_folder, device=device,
        optimizer=args.optimizer, attack_model_type=attack_model_type, weight_decay_mem=args.weight_decay_mem)
    victim_pruned_model.model.load_state_dict(org_state)
    pruner = get_pruner(args.pruner_name, victim_pruned_model.model, sparsity=args.prune_sparsity, prune_iter = args.prune_iter)
    victim_pruned_model.model = pruner.compress()

    if args.defend == "slide" or args.defend == "slide_ml2":
        victim_train_dataidx = slide_window(args.dataset_name, args.model_name, args.prune_epochs, victim_train_list, args.width, args.stride)
    elif args.defend == "slide_re" or args.defend == "slide_re_ml2":
        victim_train_dataidx = slide_window_reverse(args.dataset_name, args.model_name, args.prune_epochs, victim_train_list, args.width, args.stride)
    elif args.defend == "random":
        victim_train_dataidx = slide_random(args.dataset_name, args.prune_epochs, victim_train_list, args.width, args.stride)
    elif args.defend == "ml2":
        victim_train_dataidx_risk, victim_train_dataidx_gen = ml2_process(args.dataset_name, args.model_name, victim_train_list, args.mem_thre)
    
    start_time = time.time()
    for i in pruner.get_prune_iterations():
        pruner.prune_iteration_start()
        best_acc = 0
        count = 0
        for epoch in range(args.prune_epochs):
            pruner.update_epoch(epoch)
            if args.defend == "":
                train_acc, train_loss = victim_pruned_model.train(victim_train_loader, f"Epoch {epoch} Prune Train")
            elif args.defend == "ml2":
                victim_train_dataset_risk = Subset(trainset, victim_train_dataidx_risk)
                victim_train_loader_risk = DataLoader(victim_train_dataset_risk, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                        pin_memory=True, worker_init_fn=seed_worker)
                victim_train_dataset_gen = Subset(trainset, victim_train_dataidx_gen)
                victim_train_loader_gen = DataLoader(victim_train_dataset_gen, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                        pin_memory=True, worker_init_fn=seed_worker)
                train_acc, train_loss = victim_pruned_model.train_defend_ml2(victim_train_loader_risk, victim_train_loader_gen, f"Epoch {epoch} Prune Train with ml2")
            elif args.defend == "slide" or args.defend == "slide_re":
                curr_epoch_idx = victim_train_dataidx[epoch]
                victim_train_dataset = Subset(trainset, curr_epoch_idx)
                victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     pin_memory=True, worker_init_fn=seed_worker)
                train_acc, train_loss = victim_pruned_model.train(victim_train_loader, f"Epoch {epoch} Prune Train with slide window")
            elif args.defend == "random" :
                curr_epoch_idx = victim_train_dataidx[epoch]
                victim_train_dataset = Subset(trainset, curr_epoch_idx)
                victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     pin_memory=True, worker_init_fn=seed_worker)
                train_acc, train_loss = victim_pruned_model.train(victim_train_loader, f"Epoch {epoch} Prune Train with slide random")
            elif args.defend == "slide_re_ml2" or args.defend == "slide_ml2":
                curr_epoch_idx = victim_train_dataidx[epoch]
                victim_train_dataidx_risk, victim_train_dataidx_gen = ml2_process(args.dataset_name, args.model_name, curr_epoch_idx, args.mem_thre)
                if victim_train_dataidx_gen != []:
                    victim_train_dataset_gen = Subset(trainset, victim_train_dataidx_gen)
                    victim_train_loader_gen = DataLoader(victim_train_dataset_gen, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                        pin_memory=True, worker_init_fn=seed_worker)
                    if victim_train_dataidx_risk == []:
                        train_acc, train_loss = victim_pruned_model.train_defend_ml2(train_loader_gen=victim_train_loader_gen, log_pref= f"Epoch {epoch} Prune Train")
                
                    else:
                        victim_train_dataset_risk = Subset(trainset, victim_train_dataidx_risk)
                        victim_train_loader_risk = DataLoader(victim_train_dataset_risk, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                            pin_memory=True, worker_init_fn=seed_worker)
                        train_acc, train_loss = victim_pruned_model.train_defend_ml2(victim_train_loader_risk, victim_train_loader_gen, f"Epoch {epoch} Prune Train")
                else:
                    if victim_train_dataidx_risk != []:
                        victim_train_dataset_risk = Subset(trainset, victim_train_dataidx_risk)
                        victim_train_loader_risk = DataLoader(victim_train_dataset_risk, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                            pin_memory=True, worker_init_fn=seed_worker)
                        train_acc, train_loss = victim_pruned_model.train_defend_ml2(train_loader_risk=victim_train_loader_risk, log_pref= f"Epoch {epoch} Prune Train")
                    else:
                        raise RuntimeError('Error')
            elif args.defend == "ppb":
                train_acc, train_loss = victim_pruned_model.train_defend_ppb(victim_train_loader, log_pref=f"Epoch {epoch} Victim Prune Train With PPB", defend_arg=args.defend_arg)
            elif args.defend == "adv":
                train_acc, train_loss = victim_pruned_model.train_defend_adv(victim_train_loader, victim_dev_loader, log_pref=f"Epoch {epoch} Victim Prune Train With ADV",privacy_theta=args.defend_arg)
            elif args.defend == "relaxloss":
                train_acc, train_loss = victim_pruned_model.train_defend_relaxloss(victim_train_loader, epoch, log_pref=f"Epoch {epoch} Victim Prune Train With RelaxLoss", alpha=args.defend_arg, upper=1)
            
            dev_acc, dev_loss = victim_pruned_model.test(victim_dev_loader, f"Epoch {epoch} Prune Dev")
            test_acc, test_loss = victim_pruned_model.test(victim_test_loader, f"Epoch {epoch} Prune Test")

            if dev_acc > best_acc:
                best_acc = dev_acc
                pruner.export_model(model_path=f"{pruned_model_save_folder}/best.pth", mask_path=f"{pruned_model_save_folder}/best_mask.pth")
                count = 0
            elif args.early_stop > 0:
                count += 1
                if count > args.early_stop:
                    print(f"Early Stop at Epoch {epoch}")
                    break
        with open(f'log/{args.dataset_name}_{args.model_name}/{name}.txt', 'a') as appender:
            appender.write(f"victim iter {i} best model: Accuracy {best_acc:.3f}, Loss {dev_loss:.3f}" + "\n")
    end_time = time.time()
    sum_time = end_time - start_time
    with open(f'log/{args.dataset_name}_{args.model_name}/{name}.txt', 'a') as appender:
        if args.defend == "":
            appender.write(f"Total Base defend time: {sum_time}s" + "\n")
        else:
            appender.write(f"Total {args.defend} defend time: {sum_time}s" + "\n")
    victim_prune_acc = test_acc
    
    # prune shadow models
    print("Prune Shadow Models")
    shadow_acc_list = []
    shadow_prune_acc_list = []
    for shadow_ind in range(args.shadow_num):
        print("Prune Shadow Model:", shadow_ind)
        attack_train_dataset, attack_dev_dataset, attack_test_dataset = attack_split_list[shadow_ind]
        attack_train_loader = DataLoader(attack_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                         pin_memory=True, worker_init_fn=seed_worker)
        attack_dev_loader = DataLoader(attack_dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                       pin_memory=True, worker_init_fn=seed_worker)
        attack_test_loader = DataLoader(attack_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                        pin_memory=True, worker_init_fn=seed_worker)

        # load pretrained shadow model
        shadow_model_path = f"{save_folder}/shadow_model_{shadow_ind}/best.pth"
        shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        shadow_model.load(shadow_model_path)
        test_acc, _ = shadow_model.test(attack_test_loader, f"Pretrain Shadow")
        shadow_acc = test_acc
        shadow_acc_list.append(shadow_acc)

        org_state = copy.deepcopy(shadow_model.model.state_dict())
        pruned_shadow_model_save_folder = \
            f"{save_folder}/shadow_{prune_prefix}_model_{shadow_ind}"
        if not os.path.exists(pruned_shadow_model_save_folder):
            os.makedirs(pruned_shadow_model_save_folder)

        # prune shadow models
        shadow_pruned_model = BaseModel(
            args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, lr=prune_lr,
            weight_decay=args.weight_decay, save_folder=pruned_shadow_model_save_folder, device=device,
            optimizer=args.optimizer, attack_model_type=attack_model_type, weight_decay_mem=args.weight_decay_mem)
        shadow_pruned_model.model.load_state_dict(org_state)
        pruner = get_pruner(args.pruner_name, shadow_pruned_model.model, sparsity=args.prune_sparsity, prune_iter=args.prune_iter)
        shadow_pruned_model.model = pruner.compress()

        if args.defend == "slide" or args.defend == "slide_ml2":
            attack_train_dataidx = slide_window(args.dataset_name, args.model_name, args.prune_epochs, shadow_train_list[shadow_ind], args.width, args.stride)
        elif args.defend == "slide_re" or args.defend == "slide_re_ml2":
            attack_train_dataidx = slide_window_reverse(args.dataset_name, args.model_name, args.prune_epochs, shadow_train_list[shadow_ind], args.width, args.stride)
        elif args.defend == "random" :
            attack_train_dataidx = slide_random(args.dataset_name, args.prune_epochs, shadow_train_list[shadow_ind], args.width, args.stride)
        elif args.defend == "ml2":
            attack_train_dataidx_risk, attack_train_dataidx_gen = ml2_process(args.dataset_name, args.model_name, shadow_train_list[shadow_ind], args.mem_thre)
        for k in pruner.get_prune_iterations():
            pruner.prune_iteration_start()
            best_acc = 0
            count = 0
            for epoch in range(args.prune_epochs):
                pruner.update_epoch(epoch)
                if args.defend == "":
                    train_acc, train_loss = shadow_pruned_model.train(attack_train_loader, f"Epoch {epoch} Shadow Prune Train")
                elif args.defend == "ml2":
                    attack_train_dataset_risk = Subset(trainset, attack_train_dataidx_risk)
                    attack_train_loader_risk = DataLoader(attack_train_dataset_risk, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                            pin_memory=True, worker_init_fn=seed_worker)
                    attack_train_dataset_gen = Subset(trainset, attack_train_dataidx_gen)
                    attack_train_loader_gen = DataLoader(attack_train_dataset_gen, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                            pin_memory=True, worker_init_fn=seed_worker)
                    train_acc, train_loss = shadow_pruned_model.train_defend_ml2(attack_train_loader_risk, attack_train_loader_gen, f"Epoch {epoch} Shadow Prune Train with ml2")
                elif args.defend == "slide" or args.defend == "slide_re":
                    curr_epoch_idx = attack_train_dataidx[epoch]
                    attack_train_dataset = Subset(trainset, curr_epoch_idx)
                    attack_train_loader = DataLoader(attack_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                        pin_memory=True, worker_init_fn=seed_worker)
                    train_acc, train_loss = shadow_pruned_model.train(attack_train_loader, f"Epoch {epoch} Shadow Prune Train with slide window")
                elif args.defend == "random" :
                    curr_epoch_idx = attack_train_dataidx[epoch]
                    attack_train_dataset = Subset(trainset, curr_epoch_idx)
                    attack_train_loader = DataLoader(attack_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                        pin_memory=True, worker_init_fn=seed_worker)
                    train_acc, train_loss = shadow_pruned_model.train(attack_train_loader, f"Epoch {epoch} Shadow Prune Train with slide window")
                elif args.defend == "slide_re_ml2" or args.defend == "slide_ml2":
                    curr_epoch_idx = attack_train_dataidx[epoch]
                    attack_train_dataidx_risk, attack_train_dataidx_gen = ml2_process(args.dataset_name, args.model_name, curr_epoch_idx, args.mem_thre)
                    if attack_train_dataidx_gen != []:
                        attack_train_dataset_gen = Subset(trainset, attack_train_dataidx_gen)
                        attack_train_loader_gen = DataLoader(attack_train_dataset_gen, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                            pin_memory=True, worker_init_fn=seed_worker)
                        if attack_train_dataidx_risk == []:
                            train_acc, train_loss = shadow_pruned_model.train_defend_ml2(train_loader_gen=attack_train_loader_gen, log_pref= f"Epoch {epoch} Prune Train")
                    
                        else:
                            attack_train_dataset_risk = Subset(trainset, attack_train_dataidx_risk)
                            attack_train_loader_risk = DataLoader(attack_train_dataset_risk, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                                pin_memory=True, worker_init_fn=seed_worker)
                            train_acc, train_loss = shadow_pruned_model.train_defend_ml2(attack_train_loader_risk, attack_train_loader_gen, f"Epoch {epoch} Prune Train")
                    else:
                        if attack_train_dataidx_risk != []:
                            attack_train_dataset_risk = Subset(trainset, attack_train_dataidx_risk)
                            attack_train_loader_risk = DataLoader(attack_train_dataset_risk, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                                pin_memory=True, worker_init_fn=seed_worker)
                            train_acc, train_loss = shadow_pruned_model.train_defend_ml2(train_loader_risk=attack_train_loader_risk, log_pref= f"Epoch {epoch} Prune Train")
                        else:
                            raise RuntimeError('Error')
                    
                elif args.defend == "ppb":
                    train_acc, train_loss = shadow_pruned_model.train_defend_ppb(attack_train_loader, f"Epoch {epoch} Shadow Prune Train With PPB", defend_arg=args.defend_arg)
                elif args.defend == "adv":
                    train_acc, train_loss = shadow_pruned_model.train_defend_adv(attack_train_loader, attack_dev_loader, log_pref=f"Epoch {epoch} Shadow Prune Train With ADV", privacy_theta=args.defend_arg)
                elif args.defend == "relaxloss":
                    train_acc, train_loss = shadow_pruned_model.train_defend_relaxloss(attack_train_loader, epoch, log_pref=f"Epoch {epoch} Shadow Prune Train With RelaxLoss", alpha=args.defend_arg, upper=1)
                dev_acc, dev_loss = shadow_pruned_model.test(attack_dev_loader, f"Epoch {epoch} Shadow Prune Dev")
                test_acc, test_loss = shadow_pruned_model.test(attack_test_loader, f"Epoch {epoch} Shadow Prune Test")

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    pruner.export_model(model_path=f"{pruned_shadow_model_save_folder}/best.pth", mask_path=f"{pruned_shadow_model_save_folder}/best_mask.pth")
                    count = 0
                elif args.early_stop > 0:
                    count += 1
                    if count > args.early_stop:
                        print(f"Early Stop at Epoch {epoch}")
                        break
            with open(f'log/{args.dataset_name}_{args.model_name}/{name}.txt', 'a') as appender:
                appender.write(f"shadow {shadow_ind} iter {k} best model: Accuracy {best_acc:.3f}, Loss {dev_loss:.3f}" + "\n")  
        shadow_prune_acc = test_acc
        shadow_prune_acc_list.append(shadow_prune_acc)

    #return victim_acc, victim_prune_acc, np.mean(shadow_acc_list), np.mean(shadow_prune_acc_list)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
        args.prune_epochs = 21

    print(args)
    main(args)
