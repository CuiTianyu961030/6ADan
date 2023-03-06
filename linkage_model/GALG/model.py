from layers import StackGraphAttention, InnerProductDecoder
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def gaussian_noise_layer(input_layer, std):
    """Noise sampling."""
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def dense(inputs, input_dim, output_dim, name):
    """Used to create a dense layer."""
    with tf.variable_scope(name, reuse=None):
        tf.set_random_seed(1)
        weights = tf.get_variable("weights", shape=[input_dim, output_dim],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01), dtype=tf.float32)
        bias = tf.get_variable("bias", shape=[output_dim], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        outputs = tf.add(tf.matmul(inputs, weights), bias, name='matmul')
        return outputs


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}


class GALG(Model):
    def __init__(self, placeholders, feature_length, client_list, **kwargs):
        super(GALG, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = feature_length
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.node_ids = client_list
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder', reuse=None):
            # self.noise = gaussian_noise_layer(self.inputs, 0.1)
            self.noise = self.inputs

            self.embeddings = StackGraphAttention(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden2,
                                                  adj=self.adj,
                                                  node_ids=self.node_ids,
                                                  dropout=self.dropout,
                                                  n_heads=FLAGS.n_heads,
                                                  act=tf.nn.elu,
                                                  n_layers=FLAGS.n_layers,
                                                  logging=self.logging,
                                                  name='e_dense')(self.noise)

            self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                       act=lambda x: x,
                                                       logging=self.logging)(self.embeddings)


class VGALG(Model):
    def __init__(self, placeholders, feature_length, client_list, **kwargs):
        super(VGALG, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = feature_length
        self.n_samples = len(client_list)
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.node_ids = client_list
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder'):
            self.z_mean = StackGraphAttention(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden2,
                                              adj=self.adj,
                                              node_ids=self.node_ids,
                                              dropout=self.dropout,
                                              n_heads=FLAGS.n_heads,
                                              act=tf.nn.tanh,
                                              n_layers=FLAGS.n_layers,
                                              logging=self.logging,
                                              name='e_dense_1')(self.inputs)

            self.z_log_std = StackGraphAttention(input_dim=self.input_dim,
                                                 output_dim=FLAGS.hidden2,
                                                 adj=self.adj,
                                                 node_ids=self.node_ids,
                                                 dropout=self.dropout,
                                                 n_heads=FLAGS.n_heads,
                                                 act=tf.nn.tanh,
                                                 n_layers=FLAGS.n_layers,
                                                 logging=self.logging,
                                                 name='e_dense_2')(self.inputs)

            self.embeddings = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * self.z_log_std

            self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                       act=lambda x: x,
                                                       logging=self.logging)(self.embeddings)


class Discriminator(Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.act = tf.nn.relu

    def construct(self, inputs, reuse=False):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            tf.set_random_seed(1)
            dc_den1 = tf.nn.relu(dense(inputs, FLAGS.hidden2, FLAGS.hidden3, name='dc_dense_1'))
            dc_den2 = tf.nn.relu(dense(dc_den1, FLAGS.hidden3, FLAGS.hidden1, name='dc_dense_2'))
            outputs = dense(dc_den2, FLAGS.hidden1, 1, name='dc_outputs')
            return outputs

