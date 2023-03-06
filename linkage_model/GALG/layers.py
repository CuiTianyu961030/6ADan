import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010) initialization."""
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def fingerprint_attention(inputs, weights_w, weights_b, weights_u):
    """Basic attention layer for fingerprint attention."""
    with tf.name_scope("fingerprint_attention"):
        v = tf.tanh(tf.tensordot(inputs, weights_w, axes=1) + weights_b)
        vu = tf.tensordot(v, weights_u, axes=1)
        alphas = tf.nn.softmax(vu, name='attention_value')
        outputs = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    return outputs


def service_attention(inputs, output_dim, adj, node_ids, n_heads, act, n_layers, dropout):
    """Graph attention layer for service attention."""
    inputs = tf.expand_dims(inputs, axis=0)
    for i in range(n_layers - 1):
        inputs = [attention_head(inputs, output_dim, adj, act, dropout) for _ in range(n_heads)]
        inputs = tf.concat(inputs, axis=-1)
    outputs = [attention_head(inputs, output_dim, adj, act, dropout) for _ in range(n_heads)]
    outputs = tf.add_n(outputs) / n_heads
    outputs = tf.squeeze(tf.gather(outputs, axis=1, indices=node_ids), axis=0, name='embeddings')
    return outputs


def attention_head(inputs, output_dim, adj, act, dropout=0.):
    """Attention heads in graph attention layer."""
    with tf.name_scope("service_attention"):
        inputs = tf.layers.conv1d(inputs, output_dim, 1, use_bias=False)
        f_1 = tf.layers.conv1d(inputs, 1, 1)
        f_2 = tf.layers.conv1d(inputs, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        alphas = tf.nn.softmax(tf.nn.leaky_relu(logits) + adj, name='attention_value')
        inputs = tf.nn.dropout(inputs, 1 - dropout)
        outputs = tf.contrib.layers.bias_add(tf.matmul(alphas, inputs))
        outputs = act(outputs)
    return outputs


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class StackGraphAttention(Layer):
    """Stack graph attention layer for graph encoding."""
    def __init__(self, input_dim, output_dim, adj, node_ids, dropout=0., n_heads=1, act=tf.nn.elu, n_layers=1, **kwargs):
        super(StackGraphAttention, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_w'] = weight_variable_glorot(input_dim, output_dim, name='weights_w')
            self.vars['weights_b'] = tf.Variable(tf.random_normal([output_dim], stddev=0.1, dtype=tf.float32), name='weights_b')
            self.vars['weights_u'] = tf.Variable(tf.random_normal([output_dim], stddev=0.1, dtype=tf.float32), name='weights_u')
        self.dropout = dropout
        self.adj = adj
        self.node_ids = node_ids
        self.n_heads = n_heads
        self.output_dim = output_dim
        self.act = act
        self.n_layers = n_layers

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = fingerprint_attention(x, self.vars['weights_w'], self.vars['weights_b'], self.vars['weights_u'])
        x = tf.nn.dropout(x, 1-self.dropout)
        x = service_attention(x, self.output_dim, self.adj, self.node_ids, self.n_heads, self.act, self.n_layers, self.dropout)
        return x


class InnerProductDecoder(Layer):
    """Decoder model layer for link generation."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs
