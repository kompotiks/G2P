import argparse
from g2p import G2P


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', type=str, default='پایتون')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    g2p = G2P()
    result = g2p(args.word, args.visualize)
    print('.'.join(result))
