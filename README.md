# Skip-Gram-Word-Embedding-TF
Tensorflow implementation of skip gram word embedding model for Bengali Language and can be ported on other tensorflow model

## Folder Structure
![Folder](data/tmp/tf_dir_st.png)

```python
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

```



## TF-Graph
![Graph](data/tmp/tf_word2vec_gr.png)



## TF-Session
![Session](data/tmp/tf-word2vec_sess.png)


## RUN
Define Path of raw txt file of large number of sentences ( change in line 131 at bengali_word2vec-TF.py file)
run: python bengali_word2vec-TF.py
Model saved on data/saved_model folder

To restore Model and and use for further works just change in line 85 at pretrained_word2vec.py

