# Defending Against Membership Inference Attacks on Iteratively Pruned Deep Neural Networks
Code for the paper **"Defending Against Membership Inference Attacks on Iteratively Pruned Deep Neural Networks"** in NDSS'25.

## About
Part of the codes are modified from https://github.com/Machine-Learning-Security-Lab/mia_prune.

>Reference:
Xiaoyong Yuan and Lan Zhang. Membership inference attacks and defenses in neural network pruning. In 31st USENIX Security Symposium (USENIX Security 22), 2022.

## Getting Started

### Runing Environment and Installation

We tested on Ubuntu with Python 3.6.13 and torch-gpu. We recommend installing the dependencies via virtual env management tool, e.g., Anaconda.

```
conda create -n python-pytorch python=3.6.13
conda activate python-pytorch

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install scikit-learn pandas matplotlib nni==2.3 pynvml tensorboard
```
**Note:** All dependencies of the project are in the `requirements.txt` we provide. You can also install them directly through this file after activating the virtual environment.  

### Datasets and Models
1. **Structure of Important Folders**:
   ```plaintext
   ├─ config
   ├─ data
   │   └─ datasets
   │       ├─ cifar10-data
   │       ├─ cifar100-data
   │       ├─ cinic
   │       ├─ location
   │       ├─ purchase100
   │       └─ texas
   ├─ log
   │   ├─ cifar10_resnet18
   │   ├─ location_columnfc
   │   └─ ...
   ├─ memscore
   └─ result
       ├─ cifar10_resnet18
       │   ├─ iter_pruning_0.6_slide_ml2_model
       │   └─ ...
       ├─ location_columnfc
       │   ├─ iter_prunetxt_0.6_ml2_model
       │   └─ ...
       └─ ...
   ```

   **Description**:
   (1) In the `config` folder, configuration files for different experimental settings are stored.
   (2) In the `data/datasets` folder, different downloaded datasets are stored.
   (3) In the `log` folder, the outputs of experiments under different datasets and model architectures are stored.
   (4) The mem-scores of datasets need to be stored in the `memscore` folder in CSV format. Here, we provide the mem-score that can be used directly. The mem-score can also be recalculated using the definition or as described in our paper.
   (5) The pretrained and pruned models will be saved in the `result` folder, and you can find the `.pth` model file in the folder path named the specific dataset and model architecture.
   
2. **Prepare the Datasets**: We use three image datasets: CIFAR10, CIFAR100, CINIC; three tabular datasets: Location, Texas, Purchase. For CIFAR10/CIFAR100, you can download it directly through the `torchvision` tool in code `datasets.py`. We provide download links and saving instructions for other datasets below:
   - CINIC: https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=4&isAllowed=y \
     Download and unzip the dataset and put the `train` and `test` folders into the created `data/dataset/cinic` folder.
   - Location: https://github.com/jjy1994/MemGuard/tree/master/data/location \
     Download `data_complete.npz` and put it into the `data/dataset/location` folder.          
   - Texas: https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz \
     Download and unzip `texas/100/feats.txt` and `texas/100/labels.txt` and put them into the `data/dataset/texas` folder.
   - Purchase: https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz \
     Download and unzip `dataset_purchase` and put it into the `data/dataset/purchase100` folder and rename it to `purchase100.txt`.

   It should be noted that since the training data and test data need to be manually split for tabular datasets, it is necessary to ensure that the mem-scores correspond to the split training data. Here, we provide our split Location dataset `location.pkl`, in which the training data matches the mem-scores of the corresponding data we gave.
   
3. **Models**: We use four model architectures for evaluation: ResNet18, DenseNet121, VGG16 and FC (fully connected network).For CIFAR10, CIFAR100, and CINIC, we use ResNet18, DenseNet121, and VGG16. For Texas, Location, and Purchase, we employ an FC.

## Basic Usage
1. **Step 1: Train an original model:**
   
   ```
   python pretrain_modi.py [GPU-ID] [config_path]
   ```

2. **Step 2: Prune and fine-tune the model:**

   (1) Option 1: with `Base` defense (default)    

      ```
      python prune_modi.py [GPU-ID] [config_path] --pruner_name [pruner_name] --prune_sparsity [prune_sparsity] --prune_iter [prune_iter]
      ```
   (2) Option 2: with our defenses

   - our `defend_name` in [`slide`, `slide_re`]:  
     ```
     python prune_modi.py [GPU-ID] [config_path] --pruner_name [pruner_name] --prune_sparsity [prune_sparsity] --prune_iter [prune_iter] --defend [defend_name] --stride [step_size] --width [window_width]
     ```

   - our `defend_name` is `ml2`:  
     ```
     python prune_modi.py [GPU-ID] [config_path] --pruner_name [pruner_name] --prune_sparsity [prune_sparsity] --prune_iter [prune_iter] --defend [defend_name] --weight_decay_mem [risk_reg] --mem_thre [mem_thred]
     ```

   - our `defend_name` in [`slide_ml2`, `slide_re_ml2`]:  
      ```
      python prune_modi.py [GPU-ID] [config_path] --pruner_name [pruner_name] --prune_sparsity [prune_sparsity] --prune_iter [prune_iter] --defend [defend_name] --weight_decay_mem [risk_reg] --stride [step_size] --width    
     [window_width] --mem_thre [mem_thred]
      ```
   (3) Option 3: with other defenses

   - `defend_name` in [`ppb`, `adv`, `dp`, `relaxloss`]:  

     ```
     python prune_modi.py [GPU-ID] [config_path] --pruner_name [pruner_name] --prune_sparsity [prune_sparsity] --prune_iter [prune_iter] --defend [defend_name] --defend_arg [defend_arg]
     ```

4. **Step 3: Adaptive Attack**

   (1) Option 1: attack under `Base` method:
      ```
      python mia_modi.py [GPU-ID] [config_path] --pruner_name [pruner_name] --prune_sparsity [prune_sparsity] --attacks [attacks]
      ```
   (2) Option 2: attack under our defenses:
      ```
      python mia_modi.py [GPU-ID] [config_path] --pruner_name [pruner_name] --prune_sparsity [prune_sparsity] --attacks [attacks] --defend [defend_name] --adaptive
      ```
   (3) Option 3: attack under other defenses:
      ```
      python mia_modi.py [GPU-ID] [config_path] --pruner_name [pruner_name] --prune_sparsity [prune_sparsity] --attacks [attacks] --defend [defend_name] --defend_arg [defend_arg] --adaptive
      ```

   `attacks` can be `threshold` (involves Conf, Entr, Mentr, and Hconf), `samia`, `nn`, `nn_top3`, `nn_cls`.

## Note
1. The process output, test accuracy, attack accuracy and defense time will be saved in the `log` folder.
2. `config_path` is in the `config` folder, e.g., `./config/cifar10_resnet18.json`.  
3. `pruner_name` can be `iter_pruning` (for VGG16, ResNet18, and DenseNet121) or `iter_prunetxt` (for FC model).  
4. `prune_sparsity` we set in experiment is [0.5, 0.6, 0.7].   
5. `prune_iter` we set in experiment is 5.  
6. `slide` is RSW(H->L), `slide_re` is RSW(L->H), `ml2` is RMR, `slide_ml2` is SWMR(H->L), `slide_re_ml2` is SWMR(L->H).  
7. `risk_reg` is the coefficient of risk memory regularization.  
8. `mem_thred` is the threshold of mem-score.
9. `defend_arg` is the hyper-parameters for other defenses: [1, 2, 4, 8, 16] for `ppb`, [1, 2, 4, 8] for `adv`, [0.01, 0.1, 1] for `dp`, and 1 for `relaxloss`.

## Examples for Evaluation:
Since our defense method is executed during the fine-tuning (retraining) stage of the model pruning process, and we use a model pruning method that includes five iterations in the experiments, this require five rounds of retraining operations on a normal pruned model (victim model) and five corresponding shadow pruned models, respectively, which will require more time. Therefore, in order to simplify the evaluation process, we only select one representative image dataset (e.g., CIFAR10) and one tabular dataset (e.g., Location) for evaluation. The evaluation method on the rest of the datasets (model) are the same as the example evaluation method we give, only the `config_path` needs to be changed.

- Download the data files according to the second item in "Datasets and Models" when you use tabular or CINIC datasets.
- Please run `run_shell.sh` (you may need to run `chmod +x run_shell.sh` first), which consolidates all experimental steps on CIFAR10-ResNet18 and Location-FC. Note that we only evaluate the defense performance on the 'Base' method and our three methods under the best defense settings (`slide_re` (RSW), `ml2` (RMR), `slide_ml2` (SWMR)) with the pruning rate of 0.6. To save time, we perform metric-based attack `threshold` (i.e., Conf, Entr, Mentr, and Hconf) and a representative classifier-based attack `samia` (i.e., SAMIA).

- You can continue to evaluate other defense methods (this will require more evaluation time). An example of other defense method is as follows:
   ```
   # Prune and fine-tune models with 'ppb' defense method
   python prune_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --prune_iter 5 --defend ppb --defend_arg 4

   # Adaptive attack on pruned models with 'ppb' defense method
   python mia_modi.py 0 ./config/cifar10_resnet18.json --pruner_name iter_pruning --prune_sparsity 0.6 --attacks threshold,samia --defend ppb --defend_arg 4 --adaptive
   ```
- We also provide a running script, `run_custom.sh`, for all other defense methods used in the evaluation. You can run the script directly (you may need to run `chmod +x run_custom.sh` first) to evaluate other defense methods.
  
### Interpreting the results
The test accuracy of the iteratively pruned models with different defense methods and the attack accuracy under different MIAs can be found in the corresponding files in the `log/` directory. 

For example, the evaluation results of cifar10-resnet18 pruned model with `slide_ml2` defense are stored in `log/cifar10_resnet18/iter_pruning_0.6_slide_ml2.txt`, where "Victim pruned model test accuracy" represents prediction accuracy of the pruned model, and "[attack name] attack accuracy" represents accuracy of MIAs on the pruned model. Additionally, you can check the defense time required for the current defense through "Total [defense name] defend time".

## Citation
```
@inproceedings{shang2025iteratively,
  title = {Defending Against Membership Inference Attacks on Iteratively Pruned Deep Neural Networks},
  booktitle = {Network and Distributed System Security (NDSS) Symposium},
  author={Shang, Jing and Wang, Jian and Wang, Kailun and Liu, Jiqiang and Jiang, Nan and Armanuzzaman, MD and Zhao, Ziming},
  year={2025}
}
```



