#!usr/bin/env python

import collections
import random
import numpy as np
import codecs


def sentence_split(sen):
    return sen.strip().split()


def read_data(path):
    words = []
    with codecs.open(path, "rb", encoding='utf-8') as f:
        for x in f:
            x = sentence_split(x)
            for y in x: words.append(y)
    return words


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common())
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, reverse_dictionary


def generate_batch(start_idx, batch_size, num_skips, skip_window, data):
    data_index = start_idx
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    _buffer = collections.deque(maxlen=span)
    for _ in range(span):
        _buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the _buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = _buffer[skip_window]
            labels[i * num_skips + j, 0] = _buffer[target]
        _buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


def make_data(file_path):
    words = read_data(file_path)
    words_idx, reverse_dictionary = build_dataset(words)
    del words
    return words_idx, reverse_dictionary
