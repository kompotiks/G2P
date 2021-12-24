from typing import List

import torch
import matplotlib.pyplot as plt

from .data import PersianLexicon
from .model import Encoder, Decoder
from .config import DataConfig, ModelConfig, TestConfig


def load_model(model_path, model):
    model.load_state_dict(torch.load(
        model_path,
        map_location=lambda storage, loc: storage
    ))
    model.to(TestConfig.device)
    model.eval()
    return model


class G2P(object):
    def __init__(self, lexicon_path: str,
                 graphemes: List[str],
                 hidden_size: int,
                 phonemes: List[str],
                 encoder_model_path: str,
                 decoder_model_path: str,
                 max_len: int = 50,
                 ):
        self.max_len = max_len
        self.graphemes = graphemes
        self.phonemes = phonemes
        self.ds = PersianLexicon(graphemes, phonemes, lexicon_path)

        self.encoder_model = Encoder(len(graphemes), hidden_size)
        load_model(encoder_model_path, self.encoder_model)

        self.decoder_model = Decoder(len(phonemes), hidden_size)
        load_model(decoder_model_path, self.decoder_model)

    def _clean(self, word: str):
        word = word.replace('ё', 'е')
        word = [i for i in word if i in self.graphemes]
        return word

    def __call__(self, word: str, visualize: bool = False):
        x = [0] + [self.ds.g2idx[ch] for ch in self._clean(word)] + [1]
        x = torch.tensor(x).long().unsqueeze(1)
        with torch.no_grad():
            enc = self.encoder_model(x)

        phonemes, att_weights = [], []
        x = torch.zeros(1, 1).long().to(TestConfig.device)
        hidden = torch.ones(
            1,
            1,
            ModelConfig.hidden_size
        ).to(TestConfig.device)
        t = 0
        while True:
            with torch.no_grad():
                out, hidden, att_weight = self.decoder_model(
                    x,
                    enc,
                    hidden
                )

            att_weights.append(att_weight.detach().cpu())
            max_index = out[0, 0].argmax()
            x = max_index.unsqueeze(0).unsqueeze(0)
            t += 1

            phonemes.append(self.ds.idx2p[max_index.item()])
            if max_index.item() == 1:
                phonemes.remove('<eos>')
                break
            elif t > self.max_len:
                break

        if visualize:
            att_weights = torch.cat(att_weights).squeeze(1).numpy().T
            y, x = att_weights.shape
            plt.imshow(att_weights, cmap='gray')
            plt.yticks(range(y), ['<sos>'] + list(word) + ['<eos>'])
            plt.xticks(range(x), phonemes)
            plt.savefig(f'attention/{DataConfig.language}/{word}.png')
        return phonemes
