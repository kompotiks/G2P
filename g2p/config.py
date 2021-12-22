import os
import json

import torch

cpu = torch.device('cpu')
gpu = torch.device('cuda')


class DataConfig(object):
    language = os.getenv('LANGUAGE', 'ru')
    graphemes_path = f'resources/{language}/Graphemes.json'
    phonemes_path = f'resources/{language}/Phonemes.json'
    lexicon_path = f'resources/{language}/Lexicon.json'


class ModelConfig(object):
        try:
            with open(DataConfig.graphemes_path) as f:
                graphemes_size = len(json.load(f))

            with open(DataConfig.phonemes_path) as f:
                phonemes_size = len(json.load(f))
        except:
            pass
        graphemes_size = ''
        phonemes_size = ''
        hidden_size = 128


class TrainConfig(object):
    device = gpu if torch.cuda.is_available() else cpu
    lr = 3e-4
    batch_size = 128
    epochs = int(os.getenv('EPOCHS', '100'))
    log_path = f'log/{DataConfig.language}'


class TestConfig(object):
    device = cpu
    encoder_model_path = f'models/{DataConfig.language}/encoder_e{TrainConfig.epochs:02}.pth'
    decoder_model_path = f'models/{DataConfig.language}/decoder_e{TrainConfig.epochs:02}.pth'


GRAPHEMES = [
    "<sos>",
    "<eos>",
    "ю",
    "я",
    "а",
    "д",
    "э",
    "ь",
    "у",
    "п",
    "ъ",
    "т",
    "л",
    "с",
    "г",
    "о",
    "и",
    "р",
    "ж",
    "н",
    "ч",
    "м",
    "х",
    "з",
    "ц",
    "ы",
    "в",
    "к",
    "б",
    "ш",
    "й",
    "е",
    "ф",
    "щ"
]

PHONEMES = [
    "<sos>",
    "<eos>",
    "x",
    "sj",
    "p",
    "rj",
    "d",
    "j",
    "jE",
    "tSj",
    "b",
    "StSj",
    "i",
    "n",
    "jO",
    "Z",
    "S",
    "e",
    "hrd",
    "k",
    "dj",
    "jA",
    "Zj",
    "lj",
    "vj",
    "g",
    "o",
    "t",
    "s",
    "v",
    "mj",
    "Sj",
    "m",
    "f",
    "z",
    "bj",
    "ts",
    "zj",
    "i2",
    "pj",
    "tS",
    "a",
    "r",
    "tj",
    "u",
    "jU",
    "nj",
    "l",
    "StS"
]
