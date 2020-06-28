import torch.nn.functional as F
import torch
# external loss function from loss dir


def nll_loss(output, target):
    return F.nll_loss(output, target)

def CE(output, target):
    return F.cross_entropy(output, target)
