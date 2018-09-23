import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

NAMESPACE = 'word_predictor'
TINY = 1e-6


class WordPredictor(object):
    _context_dim = 300
    _memory_dim = 900
    _stack_dimension = 2

    def __init__(self, vocab_size):
        self._vocab_size = vocab_size
        self.g = tf.Graph()
        with self.g.as_default():
            with tf.variable_scope(NAMESPACE):
                config = tf.ConfigProto(allow_soft_placement=True)
                self.sess = tf.Session(config=config)

                # Input variables
                self.sentence_vectors_fw = tf.placeholder(tf.float32, shape=(None, None, self._vocab_size),
                                                          name='sentence_vectors_inp_fw')

                # The sentence is pre-processed by a bi-GRU
                self.Wq = tf.Variable(tf.random_uniform([self._vocab_size,
                                                         self._context_dim], -0.1, 0.1))
                self.internal_projection = lambda x: tf.nn.relu(tf.matmul(x, self.Wq))
                self.sentence_int_fw = tf.map_fn(self.internal_projection, self.sentence_vectors_fw)

                self.rnn_cell_fw = rnn.MultiRNNCell(
                    [rnn.GRUCell(self._memory_dim) for _ in range(self._stack_dimension)],
                    state_is_tuple=True)
                with tf.variable_scope('fw'):
                    output_fw, _ = tf.nn.dynamic_rnn(self.rnn_cell_fw, self.sentence_int_fw, time_major=True,
                                                     dtype=tf.float32)
                self.sentence_vector = output_fw[-1]

                # Final feedforward layers
                self.Ws1 = tf.Variable(tf.random_uniform([self._memory_dim, self._context_dim], -0.1, 0.1),
                                       name='Ws1')
                self.bs1 = tf.Variable(tf.random_uniform([self._context_dim], -0.1, 0.1), name='bs1')
                self.hidden = tf.nn.relu(tf.matmul(self.sentence_vector, self.Ws1) + self.bs1)
                self.outputs = tf.nn.softmax(tf.matmul(self.hidden, tf.transpose(self.Wq, (1, 0))))

                # Loss function and training
                self.y_ = tf.placeholder(tf.float32, shape=(None, self._vocab_size), name='y_')
                self.one = tf.ones_like(self.outputs)
                self.tiny = TINY * self.one
                self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.outputs + self.tiny))

            # Clipping the gradient
            optimizer = tf.train.AdamOptimizer(1e-3)
            gvs = optimizer.compute_gradients(self.cross_entropy)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if var.name.find(NAMESPACE) != -1]
            self.train_step = optimizer.apply_gradients(capped_gvs)
            self.sess.run(tf.global_variables_initializer())

            # Adding the summaries
            tf.summary.scalar('cross_entropy', self.cross_entropy)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('./tf_train', self.sess.graph)

    def _add_identity(self, A):
        num_nodes = A.shape[0]
        identity = np.identity(num_nodes)
        return identity + A

    def __train(self, sentence_vectors, y):
        sentence_vectors = np.array(sentence_vectors)
        sentence_vectors_fw = np.transpose(sentence_vectors, (1, 0, 2))

        y = np.array(y)

        feed_dict = {}
        feed_dict.update({self.sentence_vectors_fw: sentence_vectors_fw})
        feed_dict.update({self.y_: y})

        loss, _, summary = self.sess.run([self.cross_entropy, self.train_step, self.merged], feed_dict)
        return loss, summary

    def train(self, data, epochs=20):
        for epoch in range(epochs):
            loss, _ = self.__train([data[i][0] for i in range(len(data))],
                                   [data[i][1] for i in range(len(data))])

    def __predict(self, sentence_vectors):
        sentence_vectors = np.array(sentence_vectors)
        sentence_vectors_fw = np.transpose(sentence_vectors, (1, 0, 2))

        feed_dict = {}
        feed_dict.update({self.sentence_vectors_fw: sentence_vectors_fw})
        y_batch = self.sess.run(self.outputs, feed_dict)
        return y_batch

    def predict(self, sentence_vectors):
        output = np.array(self.__predict([sentence_vectors]))[0]
        return output

    def get_loss(self, sentence_vectors, y):
        sentence_vectors = np.array(sentence_vectors)
        sentence_vectors_fw = np.transpose(sentence_vectors, (1, 0, 2))

        y = np.array(y)

        feed_dict = {}
        feed_dict.update({self.sentence_vectors_fw: sentence_vectors_fw})
        feed_dict.update({self.y_: y})

        loss = self.sess.run([self.cross_entropy], feed_dict)
        return loss[0]

    # Loading and saving functions

    def save(self, filename):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, filename)

    def load_tensorflow(self, filename):
        with self.g.as_default():
            saver = tf.train.Saver([v for v in tf.global_variables() if NAMESPACE in v.name])
            saver.restore(self.sess, filename)

    @classmethod
    def load(self, filename, vocab_size):
        model = WordPredictor(vocab_size)
        model.load_tensorflow(filename)
        return model
