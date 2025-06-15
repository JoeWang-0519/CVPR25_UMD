from __future__ import absolute_import

from .random_sampling import RandomSampling
from .ours_two_stages import ProposedSampling

__factory = {
    'random': RandomSampling,
    'proposed_sampling': ProposedSampling
}

def names():
    return sorted(__factory.keys())

def build_strategy(method, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
    if method not in __factory:
        raise KeyError("Unknown query strategy:", method)
    return __factory[method](model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)