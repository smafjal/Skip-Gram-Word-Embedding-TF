#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import data_reader as reader
from matplotlib.font_manager import FontProperties
import cPickle as pickle


def word_embedding_model(reverse_dictionary, words_idx, ckpt_dir):
    # Model Build  Paremetars
    vocabulary_size = len(reverse_dictionary)
    batch_size = 128
    embedding_size = 50  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = min(vocabulary_size, 100)  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64  # Number of negative examples to sample.

    # Train Parametars
    num_steps = 9000  # training epoch
    display_step = 100  # show log on terminal
    demo_step = 100  # demo show after this itr
    save_step = 1000  # saved after this itr

    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.name_scope("embedding"):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                                             labels=train_labels, inputs=embed,
                                             num_sampled=num_sampled, num_classes=vocabulary_size))

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        init.run()
        saver = tf.train.Saver()
        average_loss = 0
        start_idx = 0;
        for step in range(1, num_steps):
            batch_inputs, batch_labels = reader.generate_batch(start_idx, batch_size, num_skips, skip_window, words_idx)
            start_idx = (start_idx + batch_size) % len(words_idx)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % display_step == 0:
                average_loss /= display_step
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            if step % demo_step == 0:
                sim = similarity.eval()
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
            if step % save_step == 0:
                saver.save(session, ckpt_dir + "/model.ckpt", global_step=step)

        saver.save(session, ckpt_dir + "/final-model.ckpt", global_step=num_steps)
        final_embeddings = normalized_embeddings.eval()
    return final_embeddings


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
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
    ckpt_dir = "data/saved_model"

    words_idx, reverse_dictionary = reader.make_data(raw_txt_file)
    # save pickle
    save_pickle(words_idx, words_idx_pickle)
    save_pickle(reverse_dictionary, reverse_dictionary_pickle)

    print "Dictonary-Size: ", len(reverse_dictionary)

    # do word embedding
    final_embeddings = word_embedding_model(reverse_dictionary, words_idx, ckpt_dir)

    # view model on tsne
    tsn_view(final_embeddings, reverse_dictionary)


def main():
    do_embedding()


if __name__ == "__main__":
    main()
