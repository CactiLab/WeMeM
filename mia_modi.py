"""
This code is modified from https://github.com/Machine-Learning-Security-Lab/mia_prune
"""
import argparse
import json
import pickle
import random
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from attackers import MiaAttack
from base_model import BaseModel
from datasets import get_dataset
from torch.utils.data import ConcatDataset, DataLoader, Subset


parser = argparse.ArgumentParser(description='Membership inference Attacks on Network Pruning')
parser.add_argument('device', default=0, type=int, help="GPU id to use")
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--dataset_name', default='cifar10', type=str)
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--image_size', default=32, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--early_stop', default=5, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--pruner_name', default='l1unstructure', type=str, help="prune method for victim model")
parser.add_argument('--prune_sparsity', default=0.6, type=float, help="prune sparsity for victim model")
parser.add_argument('--adaptive', action='store_true', help="adaptive attack")
parser.add_argument('--shadow_num', default=5, type=int)
parser.add_argument('--defend', default='', type=str)
parser.add_argument('--defend_arg', default=4, type=float)
parser.add_argument('--attacks', default="samia", type=str)
parser.add_argument('--original', action='store_true', help="original=true, then launch attack against original model")


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    if args.defend == "" or args.defend == "ppb" or args.defend == "adv" or args.defend == "dp" or args.defend == "relaxloss":
        prune_prefix = f"{args.pruner_name}_{args.prune_sparsity}" \
                    f"{'_' + args.defend if args.defend else ''}{'_' + str(args.defend_arg) if args.defend else ''}"
        prune_prefix2 = f"{args.pruner_name}_{args.prune_sparsity}" \
                        f"{'_' + args.defend if args.adaptive else ''}{'_' + str(args.defend_arg) if args.adaptive else ''}"
    else: 
        prune_prefix = f"{args.pruner_name}_{args.prune_sparsity}" \
                    f"{'_' + args.defend if args.defend else ''}"
        prune_prefix2 = f"{args.pruner_name}_{args.prune_sparsity}" \
                        f"{'_' + args.defend if args.adaptive else ''}"

    save_folder = f"result/{args.dataset_name}_{args.model_name}"
    if args.defend == "slide" or args.defend == "slide_re" or args.defend == "slide_re_ml2" or args.defend == "slide_ml2":
        name = f'{args.pruner_name}_{args.prune_sparsity}_{args.defend}'
    else:
        name = f'{args.pruner_name}_{args.prune_sparsity}_{args.defend}'
    print(f"Save Folder: {save_folder}")

    # Load datasets
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    if testset is None:
        total_dataset = trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
    total_size = len(total_dataset)
    data_path = f"{save_folder}/data_prepare.pkl"
    with open(data_path, 'rb') as f:
        victim_train_list, victim_train_dataset, victim_dev_dataset, victim_test_dataset, attack_split_list, shadow_train_list \
            = pickle.load(f)
    print(f"Total Data Size: {total_size}, "
          f"Victim Train Size: {len(victim_train_dataset)}, "
          f"Victim Test Size: {len(victim_test_dataset)}")
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=False)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=False)

    # Load pruned victim model
    victim_model_save_folder = save_folder + "/victim_model"
    victim_model_path = f"{victim_model_save_folder}/best.pth"
    victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    victim_model.load(victim_model_path)

    pruned_model_save_folder = f"{save_folder}/{prune_prefix}_model"
    print(f"Load Pruned Model from {pruned_model_save_folder}")
    victim_pruned_model = BaseModel(
        args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, save_folder=pruned_model_save_folder,
        device=device)
    victim_pruned_model.model.load_state_dict(torch.load(f"{pruned_model_save_folder}/best.pth"))
    victim_pruned_model.test(victim_train_loader, "Victim Pruned Model Train")
    test_acc, loss = victim_pruned_model.test(victim_test_loader, "Victim Pruned Model Test")
    with open(f'log/{args.dataset_name}_{args.model_name}/{name}.txt', 'a') as appender:
        appender.write(f"Victim pruned model test accuracy: {test_acc:.3f}" + "\n")

    # Load pruned shadow models
    shadow_model_list, shadow_prune_model_list, shadow_train_loader_list, shadow_test_loader_list = [], [], [], []
    for shadow_ind in range(args.shadow_num):
        shadow_train_dataset, shadow_dev_dataset, shadow_test_dataset = attack_split_list[shadow_ind]
        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                         pin_memory=False)
        shadow_dev_loader = DataLoader(shadow_dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                       pin_memory=False)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                        pin_memory=False)

        shadow_model_path = f"{save_folder}/shadow_model_{shadow_ind}/best.pth"
        shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
        shadow_model.load(shadow_model_path)
        pruned_shadow_model_save_folder = f"{save_folder}/shadow_{prune_prefix2}_model_{shadow_ind}"
        print(f"Load Pruned Shadow Model From {pruned_shadow_model_save_folder}")
        shadow_pruned_model = BaseModel(
            args.model_name, num_cls=args.num_cls, input_dim=args.input_dim,
            save_folder=pruned_shadow_model_save_folder, device=device)
        shadow_pruned_model.model.load_state_dict(torch.load(f"{pruned_shadow_model_save_folder}/best.pth"))
        shadow_pruned_model.test(shadow_train_loader, "Shadow Pruned Model Train")
        shadow_pruned_model.test(shadow_test_loader, "Shadow Pruned Model Test")

        shadow_model_list.append(shadow_model)
        shadow_prune_model_list.append(shadow_pruned_model)
        shadow_train_loader_list.append(shadow_train_loader)
        shadow_test_loader_list.append(shadow_test_loader)

    print("Start Membership Inference Attacks")

    if args.original:
        attack_original = True
    else:
        attack_original = False
    attacker = MiaAttack(
        victim_model, victim_pruned_model, victim_train_loader, victim_test_loader,
        shadow_model_list, shadow_prune_model_list, shadow_train_loader_list, shadow_test_loader_list,
        num_cls=args.num_cls, device=device, batch_size=args.batch_size,
        attack_original=attack_original)

    attacks = args.attacks.split(',')

    if "samia" in attacks:
        nn_trans_acc = attacker.nn_attack("nn_sens_cls", model_name="transformer")
        print(f"SAMIA attack accuracy {nn_trans_acc:.3f}")
        with open(f'log/{args.dataset_name}_{args.model_name}/{name}.txt', 'a') as appender:
            appender.write(f"SAMIA attack accuracy: {nn_trans_acc:.3f}" + "\n")

    if "threshold" in attacks:
        conf, xent, mentr, top1_conf = attacker.threshold_attack()
        print(f"Conf attack accuracy: {conf:.3f}")
        print(f"Entr attack accuracy: {xent:.3f}")
        print(f"Mentr attack accuracy: {mentr:.3f}")
        print(f"Hconf attack accuracy: {top1_conf:.3f}")
        with open(f'log/{args.dataset_name}_{args.model_name}/{name}.txt', 'a') as appender:
            appender.write(f"Conf attack accuracy: {conf:.3f}" + "\n")
            appender.write(f"Entr attack accuracy: {xent:.3f}" + "\n")
            appender.write(f"Mentr attack accuracy: {mentr:.3f}" + "\n")
            appender.write(f"Hconf attack accuracy: {top1_conf:.3f}" + "\n\n")

    if "nn" in attacks:
        nn_acc = attacker.nn_attack("nn")
        print(f"NN attack accuracy {nn_acc:.3f}")
        with open(f'log/{args.dataset_name}_{args.model_name}/{name}.txt', 'a') as appender:
            appender.write(f"NN attack accuracy {nn_acc:.3f}" + "\n")

    if "nn_top3" in attacks:
        nn_top3_acc = attacker.nn_attack("nn_top3")
        print(f"Top3-NN Attack Accuracy {nn_top3_acc}")
        with open(f'log/{args.dataset_name}_{args.model_name}/{name}.txt', 'a') as appender:
            appender.write(f"Top3-NN Attack Accuracy {nn_top3_acc}" + "\n")

    if "nn_cls" in attacks:
        nn_cls_acc = attacker.nn_attack("nn_cls")
        print(f"NNCl Attack Accuracy {nn_cls_acc}")
        with open(f'log/{args.dataset_name}_{args.model_name}/{name}.txt', 'a') as appender:
            appender.write(f"NNCl Attack Accuracy {nn_cls_acc}" + "\n")
            appender.write("\n\n")

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config_path) as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    print(args)
    main(args)
