import copy
import nltk
import numpy as np

is_subjective = [1., 0.]
is_objective = [0., 1.]


def get_words(text):
    tokenizer = nltk.tokenize.TweetTokenizer()
    words = tokenizer.tokenize(text.lower())
    return words


def get_sorted_unique_words_list(text):
    tokenizer = nltk.tokenize.TweetTokenizer()
    words = tokenizer.tokenize(text.lower())
    return sorted(list(set(words)))


class Words2OneHot:
    def __init__(self, words_list):
        self._words = sorted(list(set(words_list)))
        self._zero_vector = np.zeros(len(self._words))

    def get_length(self):
        return len(self._words)

    def __getitem__(self, word):
        one_hot = copy.deepcopy(self._zero_vector)
        try:
            one_hot[self._words.index(word)] = 1
            return one_hot
        except:
            return self._zero_vector


def get_chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def bin_data_into_buckets(data, batch_size):
    buckets = []
    size_to_data_dict = {}
    for item in data:
        sentence_length = len(item[0])
        try:
            size_to_data_dict[sentence_length].append(item)
        except:
            size_to_data_dict[sentence_length] = [item]
    for key in size_to_data_dict.keys():
        data = size_to_data_dict[key]
        chunks = get_chunks(data, batch_size)
        for chunk in chunks:
            buckets.append(chunk)
    return buckets


def get_data_from_text(words_text, bptt):
    data = []
    for i in range(len(words_text) - bptt):
        context_words = words_text[i:i + bptt - 1]
        word_to_predict = words_text[i + bptt]
        data.append((context_words, word_to_predict))
    return data


def train(data, model, words_to_one_hot, saving_dir, prefix, num_samples, epochs=20, bucket_size=10, trace_every=None):
    import sys
    import random

    buckets = bin_data_into_buckets(data, bucket_size)

    buckets = sorted(buckets, key=lambda x: random.random())
    buckets = buckets[:int(num_samples / bucket_size)]

    for i in range(epochs):
        random_buckets = sorted(buckets, key=lambda x: random.random())
        if trace_every:
            sys.stderr.write('--------- Epoch ' + str(i) + ' ---------\n')
        for bucket in random_buckets:
            training_bucket = []
            for item in bucket:
                try:
                    context_words = item[0]
                    word_to_predict = item[1]
                    context_vectors = [words_to_one_hot[item] for item in context_words]
                    y = words_to_one_hot[word_to_predict]
                    training_bucket.append((context_vectors, y))
                except Exception as e:
                    print('Exception caught during training: ' + str(e))
            if len(training_bucket) > 0:
                model.train(training_bucket, epochs=1)
        if trace_every and i % trace_every == 0:
            save_filename = saving_dir + '/' + prefix + str(i) + '.tf'
            sys.stderr.write('Saving into ' + save_filename + '\n')
            model.save(save_filename)


def get_perplexity(data, model, words_to_one_hot):
    total_loss = 0
    data_length = 0
    bucket_size = 50
    buckets = bin_data_into_buckets(data, bucket_size)
    for bucket in buckets:
        context_vectors_list = []
        y_list = []
        for item in bucket:
            data_length += 1
            context_words = item[0]
            word_to_predict = item[1]
            context_vectors_list.append([words_to_one_hot[item] for item in context_words])
            y_list.append(words_to_one_hot[word_to_predict])
        total_loss += model.get_loss(context_vectors_list, y_list)
        # print(total_loss / data_length, np.exp(total_loss / data_length))
    return total_loss / data_length, np.exp(total_loss / data_length)
