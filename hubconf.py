dependencies = ['torch', 'tqdm', 'matplotlib']
from g2p import G2P
import torch


def g2p_ru():
    hub_dir = torch.hub.get_dir()
    g2p = G2P(hub_dir)
    return g2p
