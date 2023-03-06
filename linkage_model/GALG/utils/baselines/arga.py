from extractor import Extractor
from metrics import Metric
from tracker import Tracker
import networkx as nx
import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import inspect
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

flags = tf.app.flags
FLAGS = flags.FLAGS

model_str = 'arga_vae'
data_path = 'data/cstnet.json'
max_attribute_nb = 8
max_attribute_length = 50

_LAYER_UIDS = {}

flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', .5*0.005, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 1000, 'number of iterations.')


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


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, node_ids=None, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.node_ids = node_ids

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = tf.gather(self.act(x), axis=0, indices=self.node_ids) if self.node_ids is not None else self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
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

    def fit(self):
        pass

    def predict(self):
        pass


class ARGA(Model):
    def __init__(self, placeholders, num_features, features_nonzero, client_list, **kwargs):
        super(ARGA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.client_list = client_list
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder', reuse=None):
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.elu,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1')(self.inputs)

            self.noise = gaussian_noise_layer(self.hidden1, 0.1)

            self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                               output_dim=FLAGS.hidden2,
                                               adj=self.adj,
                                               act=tf.nn.elu,
                                               dropout=self.dropout,
                                               logging=self.logging,
                                               node_ids=self.client_list,
                                               name='e_dense_2')(self.noise)

            self.z_mean = self.embeddings

            self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                       act=lambda x: x,
                                                       logging=self.logging)(self.embeddings)


class ARVGA(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, client_list, **kwargs):
        super(ARVGA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.client_list = client_list
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder'):
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.elu,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1')(self.inputs)

            self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=tf.nn.elu,
                                           dropout=self.dropout,
                                           logging=self.logging,
                                           node_ids=self.client_list,
                                           name='e_dense_2')(self.hidden1)

            self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                              output_dim=FLAGS.hidden2,
                                              adj=self.adj,
                                              act=tf.nn.tanh,
                                              dropout=self.dropout,
                                              logging=self.logging,
                                              node_ids=self.client_list,
                                              name='e_dense_3')(self.hidden1)

            self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

            self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                       act=lambda x: x,
                                                       logging=self.logging)(self.z)
            self.embeddings = self.z


class Discriminator(Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.act = tf.nn.relu

    def construct(self, inputs, reuse=False):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # np.random.seed(1)
            tf.set_random_seed(1)
            dc_den1 = tf.nn.relu(dense(inputs, FLAGS.hidden2, FLAGS.hidden3, name='dc_den1'))
            dc_den2 = tf.nn.relu(dense(dc_den1, FLAGS.hidden3, FLAGS.hidden1, name='dc_den2'))
            output = dense(dc_den2, FLAGS.hidden1, 1, name='dc_output')
            return output


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm, d_real, d_fake):
        preds_sub = preds
        labels_sub = labels

        self.real = d_real

        # Discrimminator Loss
        self.dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real), logits=self.real, name='dclreal'))

        self.dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake, name='dcfake'))
        self.dc_loss = self.dc_loss_fake + self.dc_loss_real

        # Generator loss
        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake, name='gl'))

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.generator_loss = generator_loss + self.cost
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        all_variables = tf.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.name]
        en_var = [var for var in all_variables if 'e_' in var.name]

        with tf.variable_scope(tf.get_variable_scope()):
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                                  beta1=0.9, name='adam1').minimize(self.dc_loss,
                                                                                                    var_list=dc_var)  # minimize(dc_loss_real, var_list=dc_var)

            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                              beta1=0.9, name='adam2').minimize(self.generator_loss,
                                                                                                var_list=en_var)

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, d_real, d_fake):
        preds_sub = preds
        labels_sub = labels

        # Discrimminator Loss
        dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
        dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
        self.dc_loss = dc_loss_fake + dc_loss_real

        # Generator loss
        self.generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        all_variables = tf.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.op.name]
        en_var = [var for var in all_variables if 'e_' in var.op.name]

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                                  beta1=0.9, name='adam1').minimize(self.dc_loss,
                                                                                                    var_list=dc_var)  # minimize(dc_loss_real, var_list=dc_var)

            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                              beta1=0.9, name='adam2').minimize(self.generator_loss,
                                                                                                var_list=en_var)

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)


def fingerprint_attention(inputs, weights_w, weights_b, weights_u):
    """Basic attention layer for fingerprint attention."""
    with tf.name_scope("fingerprint_attention"):
        v = tf.tanh(tf.tensordot(inputs, weights_w, axes=1) + weights_b)
        vu = tf.tensordot(v, weights_u, axes=1)
        alphas = tf.nn.softmax(vu, name='attention_value')
        outputs = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    return outputs


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        # np.random.seed(1)
        tf.set_random_seed(1)
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def get_placeholder(adj):
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'real_distribution': tf.placeholder(dtype=tf.float32, shape=[adj.shape[0], FLAGS.hidden2],
                                            name='real_distribution')
    }
    return placeholders


def get_model(model_str, placeholders, num_features, num_nodes, features_nonzero, client_list):
    discriminator = Discriminator()
    d_real = discriminator.construct(placeholders['real_distribution'])
    model = None
    if model_str == 'arga_ae':
        model = ARGA(placeholders, num_features, features_nonzero, client_list)

    elif model_str == 'arga_vae':
        model = ARVGA(placeholders, num_features, num_nodes, features_nonzero, client_list)

    return d_real, discriminator, model


def format_data(adj, features, server_hits, labels):
    raw_adj = adj
    adj = nx.adjacency_matrix(nx.from_numpy_matrix(adj))

    adj_orig, train_adj, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false, client_list = \
        build_test_edges(labels)

    adj_orig.eliminate_zeros()

    features = np.array([preprocess_features(feature) for feature in features]).reshape(features.shape[0],
                                                                                        max_attribute_nb * max_attribute_length)
    features = sp.csr_matrix(features)

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    adj_norm = preprocess_graph(adj)

    num_nodes = train_adj.shape[0]

    pos_weight = float(train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) / train_adj.sum()
    norm = train_adj.shape[0] * train_adj.shape[0] / float((train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) * 2)

    adj_label = train_adj + sp.eye(train_adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    items = [adj, num_nodes, pos_weight, norm, adj_norm, adj_label, features, train_adj, train_edges, train_edges_false,
             val_edges, val_edges_false, test_edges, test_edges_false, adj_orig, client_list, num_features, features_nonzero]
    feas = {}
    for item in items:
        feas[retrieve_name(item)] = item

    return feas


def build_test_edges(labels):
    positive_edges = []
    for i in range(1, int(np.max(labels)) + 1):
        one_user_list = np.argwhere(labels == i)
        one_user_edges = create_tuple_edges(one_user_list)
        positive_edges = positive_edges + one_user_edges

    client_list = np.where(labels)[0]

    positive_edges = [[np.where(client_list == i)[0][0], np.where(client_list == j)[0][0]] for [i, j] in positive_edges]

    negative_edges = []
    while len(negative_edges) < len(positive_edges):
        index_a = np.random.randint(0, len(client_list))
        index_b = np.random.randint(0, len(client_list))
        if index_a != index_b and labels[client_list[index_a]] != labels[client_list[index_b]] and \
           [client_list[index_a], client_list[index_b]] not in negative_edges and \
           [client_list[index_b], client_list[index_a]] not in negative_edges:
            negative_edges.append([client_list[index_a], client_list[index_b]])
    negative_edges = [[np.where(client_list == i)[0][0], np.where(client_list == j)[0][0]] for [i, j] in negative_edges]

    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)

    val_nb = int(np.floor(len(negative_edges) / 20.))
    test_nb = int(np.floor(len(positive_edges) / 10.))

    val_edges = positive_edges[:val_nb]
    val_edges_false = negative_edges[:val_nb]
    test_edges = positive_edges[val_nb:val_nb+test_nb]
    test_edges_false = negative_edges[val_nb:val_nb+test_nb]
    train_edges = positive_edges[val_nb+test_nb:]
    train_edges_false = negative_edges[val_nb+test_nb:]

    train_adj = np.zeros((len(client_list), len(client_list)))
    for [i, j] in train_edges:
        train_adj[i][j] = 1
        train_adj[j][i] = 1
    train_adj = sp.csr_matrix(train_adj)

    adj_orig = np.zeros((len(client_list), len(client_list)))
    for [i, j] in positive_edges:
        adj_orig[i][j] = 1
        adj_orig[j][i] = 1
    adj_orig = sp.csr_matrix(adj_orig)

    return adj_orig, train_adj, train_edges, train_edges_false, val_edges, val_edges_false, \
           test_edges, test_edges_false, client_list


def get_optimizer(model_str, model, discriminator, placeholders, pos_weight, norm, d_real,num_nodes):
    if model_str == 'arga_ae':
        d_fake = discriminator.construct(model.embeddings, reuse=True)
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          d_real=d_real,
                          d_fake=d_fake)
    elif model_str == 'arga_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           d_real=d_real,
                           d_fake=discriminator.construct(model.embeddings, reuse=True))
    return opt


def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    z_real_dist = np.random.randn(adj.shape[0], FLAGS.hidden2)
    feed_dict.update({placeholders['real_distribution']: z_real_dist})

    for j in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)

    avg_cost = reconstruct_loss

    return emb, avg_cost


def create_tuple_edges(array):
    tuple_array = []
    temp_array = array
    for a in array:
        temp_array = np.delete(temp_array, 0)
        if len(temp_array) == 0:
            break
        for b in temp_array:
            tuple_array.append([a[0], b])
    return tuple_array


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


if __name__ == '__main__':
    # Extract data
    data_extractor = Extractor(data_path, max_attribute_nb, max_attribute_length)
    data_extractor.load_data()
    net = data_extractor.build_net()
    attributes = data_extractor.build_attribute_vec()
    distributions = data_extractor.build_distributions()
    labels = data_extractor.build_labels()

    # Format data
    feas = format_data(net, attributes, distributions, labels)

    # Define placeholders
    placeholders = get_placeholder(feas['adj_orig'])

    d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'],
                                                feas['features_nonzero'], feas['client_list'])

    opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real,
                        feas['num_nodes'])

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    linkgen_ap_history = []
    linkgen_auc_history = []
    tracking_ap_history = []
    tracking_auc_history = []

    # Train model
    for epoch in range(FLAGS.iterations):

        emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj_orig'])

        lg_train = Metric(feas['val_edges'], feas['val_edges_false'])
        auc_curr, ap_curr, _ = lg_train.get_scores(emb, feas)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost), "val_auc=",
              "{:.5f}".format(auc_curr), "val_ap=", "{:.5f}".format(ap_curr))

        if (epoch + 1) % 50 == 0:
            lg_test = Metric(feas['test_edges'], feas['test_edges_false'])
            auc_score, ap_score, _ = lg_test.get_scores(emb, feas)
            linkgen_auc_history.append(auc_score)
            linkgen_ap_history.append(ap_score)

            print('\033[92mTest scores of address correlation: auc %.5f ap %.5f\033[0m' % (auc_score, ap_score))

            tracker = Tracker(feas['test_edges'], feas['test_edges_false'], feas['train_edges'])
            tracking_auc, tracking_ap = tracker.user_tracking_scores(emb, feas)
            tracking_auc_history.append(tracking_auc)
            tracking_ap_history.append(tracking_ap)

            print('\033[93mTest scores of user tracking task: auc %.5f ap %.5f\033[0m' % (tracking_auc, tracking_ap))

    linkgen_auc_history.sort()
    linkgen_ap_history.sort()
    tracking_auc_history.sort()
    tracking_ap_history.sort()

    print('\033[94mTest max scores of address correlation: auc %.5f ap %.5f\033[0m' % (
    np.max(linkgen_auc_history), np.max(linkgen_ap_history)))
    print('\033[94mTest max scores of user tracking task: auc %.5f ap %.5f\033[0m' % (
    np.max(tracking_auc_history), np.max(tracking_ap_history)))

    print('\033[95mFinal address correlation auc: %.5f +/- %.5f\033[0m' %
          ((linkgen_auc_history[-1] + linkgen_auc_history[-3]) / 2,
           (linkgen_auc_history[-1] - linkgen_auc_history[-3]) / 2))
    print('\033[95mFinal address correlation ap: %.5f +/- %.5f\033[0m' %
          (
          (linkgen_ap_history[-1] + linkgen_ap_history[-3]) / 2, (linkgen_ap_history[-1] - linkgen_ap_history[-3]) / 2))
    print('\033[95mFinal user tracking auc: %.5f +/- %.5f\033[0m' %
          ((tracking_auc_history[-1] + tracking_auc_history[-3]) / 2,
           (tracking_auc_history[-1] - tracking_auc_history[-3]) / 2))
    print('\033[95mFinal user tracking ap: %.5f +/- %.5f\033[0m' %
          ((tracking_ap_history[-1] + tracking_ap_history[-3]) / 2,
           (tracking_ap_history[-1] - tracking_ap_history[-3]) / 2))

