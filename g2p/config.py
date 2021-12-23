import os
import json
from resources.ru_2.token import GRAPHEMES, PHONEMES
import torch

cpu = torch.device('cpu')
gpu = torch.device('cuda')


class DataConfig(object):
    language = os.getenv('LANGUAGE', 'ru_2')
    graphemes_path = f'resources/{language}/Graphemes.json'
    phonemes_path = f'resources/{language}/Phonemes.json'
    lexicon_path = f'resources/{language}/Lexicon.json'


class ModelConfig(object):
    graphemes_size = len(GRAPHEMES)
    phonemes_size = len(PHONEMES)
    hidden_size = 128


class TrainConfig(object):
    device = gpu if torch.cuda.is_available() else cpu
    lr = 3e-4
    batch_size = 128
    epochs = int(os.getenv('EPOCHS', '10'))
    log_path = f'log/{DataConfig.language}'


class TestConfig(object):
    device = cpu
    encoder_model_path = f'models/{DataConfig.language}/encoder_e{TrainConfig.epochs:02}.pth'
    decoder_model_path = f'models/{DataConfig.language}/decoder_e{TrainConfig.epochs:02}.pth'
