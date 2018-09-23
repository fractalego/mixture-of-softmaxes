import os
import random

from mixture_of_softmax.mos_model import MosWordPredictor
from mixture_of_softmax.utils import Words2OneHot
from mixture_of_softmax.utils import get_perplexity
from mixture_of_softmax.utils import get_words
from mixture_of_softmax.utils import get_data_from_text

_bucket_size = 10
_bptt = 35
_mixture_components = 15
_path = os.path.dirname(__file__)
_saving_dir = os.path.join(_path, '../data/save')
_all_tokens_filename = os.path.join(_path, '../data/penn/all.txt')
_training_filename = os.path.join(_path, '../data/penn/test.txt')

if __name__ == '__main__':
    text = open(_training_filename, encoding="ISO-8859-1").read()
    words_list = get_words(text)

    tokens_text = open(_all_tokens_filename, encoding="ISO-8859-1").read()
    tokens_list = get_words(tokens_text)
    word_to_one_hot = Words2OneHot(tokens_list)

    data = get_data_from_text(words_list, _bptt)

    i = 8
    nn_model = MosWordPredictor.load(_saving_dir + '/mos-' + str(i) + '.tf',
                                     word_to_one_hot.get_length(),
                                     mixture_components=_mixture_components)
    total_loss, ppl = get_perplexity(data, nn_model, word_to_one_hot)
    print('total_loss=', total_loss)
    print('ppl=', ppl)
