# ----------------Evaluate Other Existing Defenses on CIFAR10-ResNet18----------------
## Step 1: Pretrain original models (one victim model and five shadow models)
python pretrain_modi.py 0 ./config/cifar10_resnet18.json

## Step 2-1: Prune and fine-tune models with 'ppb' defense method
python prune_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --prune_iter 5 --defend ppb --defend_arg 4
## Step 2-2: Adaptive attack on pruned models with 'ppb' defense method
python mia_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --attacks threshold,samia --defend ppb --defend_arg 4 --adaptive

## Step 3-1: Prune and fine-tune models with 'adv' defense method
python prune_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --prune_iter 5 --defend adv --defend_arg 2
## Step 3-2: Adaptive attack on pruned models with 'adv' defense method
python mia_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --attacks threshold,samia --defend adv --defend_arg 2 --adaptive

## Step 4-1: Prune and fine-tune models with 'relaxloss' defense method
python prune_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --prune_iter 5 --defend relaxloss --defend_arg 1
## Step 4-2: Adaptive attack on pruned models with 'relaxloss' defense method
python mia_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --attacks threshold,samia --defend relaxloss --defend_arg 1 --adaptive

## Step 5-1: Prune and fine-tune models with 'dp' defense method
python prune_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --prune_iter 5 --defend dp --defend_arg 0.1
## Step 5-2: Adaptive attack on pruned models with 'dp' defense method
python mia_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --attacks threshold,samia --defend dp --defend_arg 0.1 --adaptive



# ---------------------Evaluate Other Existing Defenses on Location-FC---------------------
## Step 1: Pretrain original models (one victim model and five shadow models)
python pretrain_modi.py 0 ./config/location.json

## Step 2-1: Prune and fine-tune models with 'ppb' defense method
python prune_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --prune_iter 5 --defend ppb --defend_arg 4
## Step 2-2: Adaptive attack on pruned models with 'ppb' defense method
python mia_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --attacks threshold,samia --defend ppb --defend_arg 4 --adaptive

## Step 3-1: Prune and fine-tune models with 'adv' defense method
python prune_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --prune_iter 5 --defend adv --defend_arg 2
## Step 3-2: Adaptive attack on pruned models with 'adv' defense method
python mia_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --attacks threshold,samia --defend adv --defend_arg 2 --adaptive

## Step 4-1: Prune and fine-tune models with 'relaxloss' defense method
python prune_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --prune_iter 5 --defend relaxloss --defend_arg 1
## Step 4-2: Adaptive attack on pruned models with 'relaxloss' defense method
python mia_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --attacks threshold,samia --defend relaxloss --defend_arg 1 --adaptive

## Step 5-1: Prune and fine-tune models with 'dp' defense method
python prune_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --prune_iter 5 --defend dp --defend_arg 0.1
## Step 5-2: Adaptive attack on pruned models with 'dp' defense method
python mia_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --attacks threshold,samia --defend dp --defend_arg 0.1 --adaptive



