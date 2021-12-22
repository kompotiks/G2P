import argparse
from g2p import G2P


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', type=str, default='پایتون')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    g2p = G2P(
        lexicon_path='resources/ru/Lexicon.json',
        graphemes_size=34,
        hidden_size=128,
        phonemes_size=49,
        encoder_model_path='models/ru/encoder_e100.pth',
        decoder_model_path='models/ru/decoder_e100.pth',
    )
    result = g2p(args.word, args.visualize)
    print('.'.join(result))
