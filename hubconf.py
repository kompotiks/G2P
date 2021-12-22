dependencies = ['torch', 'tqdm', 'matplotlib']
from g2p import G2P
import torch


def g2p_ru():
    hub_dir = torch.hub.get_dir()
    g2p = G2P(
        lexicon_path=f'{hub_dir}/kompotiks_G2P_master/resources/ru/Lexicon.json',
        graphemes_size=34,
        hidden_size=128,
        phonemes_size=49,
        encoder_model_path=f'{hub_dir}kompotiks_G2P_master/models/ru/encoder_e100.pth',
        decoder_model_path=f'{hub_dir}/kompotiks_G2P_master/models/ru/decoder_e100.pth',
    )
    return g2p
