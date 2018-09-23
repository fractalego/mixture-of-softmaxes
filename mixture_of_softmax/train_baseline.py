import os
import random

from mixture_of_softmax.baseline_model import WordPredictor
from mixture_of_softmax.utils import Words2OneHot
from mixture_of_softmax.utils import train
from mixture_of_softmax.utils import get_words
from mixture_of_softmax.utils import get_data_from_text

_bucket_size = 20
_bptt = 35
_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/save')
_all_tokens_filename = os.path.join(_path, '../data/penn/all.txt')
_training_filename = os.path.join(_path, '../data/penn/train.txt')

if __name__ == '__main__':
    text = open(_training_filename, encoding="ISO-8859-1").read()
    words_list = get_words(text)

    tokens_text = open(_all_tokens_filename, encoding="ISO-8859-1").read()
    tokens_list = get_words(tokens_text)
    word_to_one_hot = Words2OneHot(tokens_list)

    data = get_data_from_text(words_list, _bptt)
    data = sorted(data, key=lambda x: random.random())

    nn_model = WordPredictor(word_to_one_hot.get_length())
    train(data, nn_model, word_to_one_hot, _saving_dir, num_samples=500000, prefix='baseline-', epochs=39, bucket_size=_bucket_size, trace_every=1)
