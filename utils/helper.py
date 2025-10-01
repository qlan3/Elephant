import os
import sys
import pickle
import torch
import psutil
import random
import datetime
import numpy as np
import jax.numpy as jnp
from jax import tree_util


def get_time_str():
    return datetime.datetime.now().strftime("%y.%m.%d-%H:%M:%S")


def rss_memory_usage():
    """
    Return the resident memory usage in MB
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2**20)
    return mem


def str_to_class(module_name, class_name):
    """
    Convert string to class
    """
    return getattr(sys.modules[module_name], class_name)


def set_random_seed(seed):
    """
    Set all random seeds
    """
    random.seed(seed)
    np.random.seed(seed)


def set_one_thread():
    """
    Set number of threads for pytorch to 1
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # torch.set_num_threads(1)


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def tree_stack(trees, axis=0):
    """
    From: https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(leaf, axis=axis) for leaf in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_transpose(list_of_trees):
    """
    Convert a list of trees of identical structure into a single tree of lists.
    Act the same as tree_stack
    """
    return tree_util.tree_map(lambda *xs: jnp.array(xs), *list_of_trees)


def tree_unstack(tree):
    """
    From: https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = tree_util.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(leaf) for leaf in new_leaves]
    return new_trees


def tree_concatenate(trees):
    """
    Adapted from tree_stack.
    Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((concatenate(a, a'), concatenate(b, b')), concatenate(c, c')).
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.concatenate(leaf) for leaf in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def save_model_param(model_param, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(model_param, f)


def load_model_param(filepath):
    with open(filepath, "rb") as f:
        model_param = pickle.load(f)
    model_param = tree_util.tree_map(jnp.array, model_param)
    return model_param


def assert_same_structure_and_shapes(x, y):
    # Check if keys match at all levels
    assert tree_util.tree_structure(x) == tree_util.tree_structure(y), (
        "Structure mismatch!"
    )

    def check_shape(leaf1, leaf2):
        assert leaf1.shape == leaf2.shape, "Shape mismatch"

    # Check if shapes match for all leaves
    tree_util.tree_map(check_shape, x, y)


def to_tensor(x, device):
    """
    Convert an array to tensor
    """
    x = torch.as_tensor(x, device=device, dtype=torch.float32)
    return x


def to_numpy(t):
    """
    Convert a tensor to numpy
    """
    return t.cpu().detach().numpy()
