import argparse
from g2p import G2P


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', type=str, default='привет')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    g2p = G2P(
        lexicon_path='resources/ru_2/Lexicon.json',
        graphemes_size=39,
        hidden_size=128,
        phonemes_size=50,
        encoder_model_path='models/ru_2/encoder_e10.pth',
        decoder_model_path='models/ru_2/decoder_e10.pth',
    )
    result = g2p(args.word, args.visualize)
    print('.'.join(result))
