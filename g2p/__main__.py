import argparse
from g2p import G2P


def g2p_ru_1():
    from resources.ru_1.token import GRAPHEMES, PHONEMES
    g2p = G2P(
        lexicon_path='resources/ru_1/Lexicon.json',
        graphemes=GRAPHEMES,
        hidden_size=128,
        phonemes=PHONEMES,
        encoder_model_path='models/ru_1/encoder_e100.pth',
        decoder_model_path='models/ru_1/decoder_e100.pth',
    )
    return g2p


def g2p_ru_2():
    from resources.ru_2.token import GRAPHEMES, PHONEMES
    g2p = G2P(
        lexicon_path='resources/ru_2/Lexicon.json',
        graphemes=GRAPHEMES,
        hidden_size=128,
        phonemes=PHONEMES,
        encoder_model_path='models/ru_2/encoder_e10.pth',
        decoder_model_path='models/ru_2/decoder_e10.pth',
    )
    return g2p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', type=str, default='привет')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    g2p = g2p_ru_1()
    result = g2p(args.word)
    print('.'.join(result))
