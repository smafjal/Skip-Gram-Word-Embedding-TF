# Skip-Gram-Word-Embedding-TF
Tensorflow implementation of skip gram word embedding model for Bengali Language and can be ported on other tensorflow model

## Folder Structure
![Folder](data/tmp/tf_dir_st.png)

## TF-Graph
![Graph](data/tmp/tf_word2vec_gr.png)
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


## TF-Session
![Session](data/tmp/tf-word2vec_sess.png)
```python
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
```


## RUN
 1. Define Path of raw txt file of large number of sentences ( change in line 131 at bengali_word2vec-TF.py file)
 2. run: python bengali_word2vec-TF.py
 3. Model saved on data/saved_model folder
 4. To restore Model and and use for further works just change in line 85 at pretrained_word2vec.py

