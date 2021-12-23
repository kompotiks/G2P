dependencies = ['torch', 'tqdm', 'matplotlib']
from g2p import G2P
import torch


def g2p_ru_1():
    from resources.ru_1.token import GRAPHEMES, PHONEMES
    hub_dir = torch.hub.get_dir()
    g2p = G2P(
        lexicon_path=f'{hub_dir}/kompotiks_G2P_master/resources/ru_1/Lexicon.json',
        graphemes=GRAPHEMES,
        hidden_size=128,
        phonemes=PHONEMES,
        encoder_model_path=f'{hub_dir}/kompotiks_G2P_master/models/ru_1/encoder_e100.pth',
        decoder_model_path=f'{hub_dir}/kompotiks_G2P_master/models/ru_1/decoder_e100.pth',
    )
    return g2p


def g2p_ru_2():
    from resources.ru_2.token import GRAPHEMES, PHONEMES
    hub_dir = torch.hub.get_dir()
    g2p = G2P(
        lexicon_path=f'{hub_dir}/kompotiks_G2P_master/resources/ru_2/Lexicon.json',
        graphemes=GRAPHEMES,
        hidden_size=128,
        phonemes=PHONEMES,
        encoder_model_path=f'{hub_dir}/kompotiks_G2P_master/models/ru_2/encoder_e10.pth',
        decoder_model_path=f'{hub_dir}/kompotiks_G2P_master/models/ru_2/decoder_e10.pth',
    )
    return g2p
