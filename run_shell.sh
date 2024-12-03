# ----------------Evaluation CIFAR10-ResNet18----------------
## Step 1: Pretrain original models (one victim model and five shadow models)
python pretrain_modi.py 0 ./config/cifar10_resnet18.json

## Step 2-1: Prune and fine-tune models with 'Base' defense method
python prune_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --prune_iter 5
## Step 2-2: Adaptive attack on pruned models with 'Base' defense method
python mia_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --attacks threshold,samia

## Step 3-1: Prune and fine-tune models with 'slide_re' defense method (Ours defense)
python prune_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --prune_iter 5 --defend slide_re --stride 50 --width 500
## Step 3-2: Adaptive attack on pruned models with 'slide_re' defense method (Ours defense)
python mia_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --attacks threshold,samia --defend slide_re --adaptive

## Step 4-1: Prune and fine-tune models with 'ml2' defense method (Ours defense)
python prune_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --prune_iter 5 --defend ml2 --weight_decay_mem 0.1 --mem_thre 0.5
## Step 4-2: Adaptive attack on pruned models with 'ml2' defense method (Ours defense)
python mia_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --attacks threshold,samia --defend ml2 --adaptive

## Step 5-1: Prune and fine-tune models with 'slide_ml2' defense method (Ours defense)
python prune_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --prune_iter 5 --defend slide_ml2 --weight_decay_mem 0.1 --stride 100 --width 500 --mem_thre 0.5
## Step 5-2: Adaptive attack on pruned models with 'slide_ml2' defense method (Ours defense)
python mia_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --attacks threshold,samia --defend slide_ml2 --adaptive




# ---------------------Evaluation Location-FC---------------------
## Step 1: Pretrain original models (one victim model and five shadow models)
python pretrain_modi.py 0 ./config/location.json

## Step 2-1: Prune and fine-tune models with 'Base' defense method
python prune_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --prune_iter 5
## Step 2-2: Adaptive attack on pruned models with 'Base' defense method
python mia_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --attacks threshold,samia

## Step 3-1: Prune and fine-tune models with 'slide_re' defense method (Ours defense)
python prune_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --prune_iter 5 --defend slide_re --stride 1 --width 15
## Step 3-2: Adaptive attack on pruned models with 'slide_re' defense method (Ours defense)
python mia_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --attacks threshold,samia --defend slide_re --adaptive

## Step 4-1: Prune and fine-tune models with 'ml2' defense method (Ours defense)
python prune_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --prune_iter 5 --defend ml2 --weight_decay_mem 0.1 --mem_thre 0.6
## Step 4-2: Adaptive attack on pruned models with 'ml2' defense method (Ours defense)
python mia_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --attacks threshold,samia --defend ml2 --adaptive

## Step 5-1: Prune and fine-tune models with 'slide_ml2' defense method (Ours defense)
python prune_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --prune_iter 5 --defend slide_ml2 --weight_decay_mem 0.1 --stride 1 --width 15 --mem_thre 0.6
## Step 5-2: Adaptive attack on pruned models with 'slide_ml2' defense method (Ours defense)
python mia_modi.py 0 ./config/location.json --pruner_name iter_prunetxt --prune_sparsity 0.6 --attacks threshold,samia --defend slide_ml2 --adaptive




