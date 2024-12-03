from nni.algorithms.compression.pytorch.pruning import (LevelPruner, LotteryTicketPruner)


def get_pruner(pruner_name, model, sparsity=0.5, prune_iter = 5):
    if pruner_name == "l1unstructure":
        config_list = [{
            'sparsity': sparsity,
            'op_types': ["default"]
        }]
        return LevelPruner(model, config_list)
    elif pruner_name == "iter_pruning":
        config_list = [{
            'prune_iterations': prune_iter - 1,
            'sparsity': sparsity,
            'op_types': ['default']
        }]
        return LotteryTicketPruner(model, config_list, reset_weights=False)
    elif pruner_name == "iter_prunetxt":
        config_list = [{
            'prune_iterations': prune_iter - 1,
            'sparsity': sparsity,
            'op_names': ['fc1', 'fc2']
        }]
        return LotteryTicketPruner(model, config_list, reset_weights=False)
    else:
        raise ValueError

