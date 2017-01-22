#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.font_manager import FontProperties
import cPickle as pickle


def word_embedding_model(reverse_dictionary, ckpt_file):
    vocabulary_size = len(reverse_dictionary)
    embedding_size = 50
    valid_size = 16
    valid_window = min(vocabulary_size, 100)
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    graph = tf.Graph()
    with graph.as_default():
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        with tf.name_scope("embedding"):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        init.run()
        saver = tf.train.Saver()
        saver.restore(session, ckpt_file)
        final_embeddings = normalized_embeddings.eval()

        # similarity findings
        sim = session.run(similarity)
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Nearest of { %s }\n" % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = "%s %s," % (log_str, close_word)
            print(log_str)
        print "=" * 80

    return final_embeddings


def plot_with_labels(low_dim_embs, labels, filename='tsne-pretrained.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(25, 20))  # in inches
    bd_font = FontProperties(fname='/home/afjal/.local/share/fonts/SolaimanLipi_20-04-07.ttf')
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontproperties=bd_font)
    plt.savefig(filename)


def tsn_view(final_embeddings, reverse_dictionary):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 50
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)


def save_pickle(data, path):
    pickle.dump(data, open(path, "wb"))


def load_pickle(path):
    data = pickle.load(open(path, "rb"))
    return data


def do_embedding():
    raw_txt_file = "data/raw_data/corpus_bd_02.txt"
    words_idx_pickle = "data/pickle_data/words_idx.pickle"
    reverse_dictionary_pickle = "data/pickle_data/reverse_dictionary.pickle"
    ckpt_file = "data/saved_model/final-model.ckpt-9000"

    reverse_dictionary = load_pickle(reverse_dictionary_pickle)
    print "Dictonary-Size: ", len(reverse_dictionary)

    final_embeddings = word_embedding_model(reverse_dictionary, ckpt_file)
    tsn_view(final_embeddings, reverse_dictionary)


def main():
    do_embedding()


if __name__ == "__main__":
    main()
