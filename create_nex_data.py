import pandas as pd


def create_lexicon():
    data = []
    for word, phonem in zip(words, phonems):
        data.append([word, phonem])

    with open('resources/my/Lexicon.json', 'w', encoding='utf-8') as f:
        f.write('[\n')
        for sample in data:
            f.write('\t[\n')
            f.write('\t\t"' + sample[0] + '",\n')
            f.write('\t\t"' + sample[1] + '"\n')
            f.write('\t],\n')
        f.write(']\n')


def unique_phonemes(phonemes: list):
    phonem_unique = sorted(list(set(' '.join(phonemes).split(' '))))

    with open('resources/ru/Phonemes.json', 'w', encoding='utf-8') as f:
        f.write(str(phonem_unique).replace("'", '"'))


def unique_graphemes(graphemes: list):
    graphem_unique = sorted(list(set(''.join(graphemes))))

    with open('resources/ru/Graphemes.json', 'w', encoding='utf-8') as f:
        f.write(str(graphem_unique).replace("'", '"'))


if __name__ == '__main__':
    df = pd.read_csv('lexicon.txt', sep='\t', header=None)
    words, phonems = df[0].to_list(), df[1].to_list()