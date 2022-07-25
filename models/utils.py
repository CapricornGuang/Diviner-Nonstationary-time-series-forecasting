import torch
import torch.nn as nn

def get_module_list(module, n, lists=None):
    if lists is None:
        return nn.ModuleList([module for i in range(n)])
    elif lists is not None and type(lists) is type(torch.nn.ModuleList()):
        for i in range(n):
            lists.append(module)
        return lists

def get_keepdim_paddings(kernel_size):
    return (kernel_size-1)//2

def is_odd(n):
    return type(n) is int and n%2==1