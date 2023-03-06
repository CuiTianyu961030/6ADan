from model import GALG, VGALG, Discriminator
from optimizer import OptimizerAE, OptimizerVAE
from sklearn.decomposition import PCA
import scipy.sparse as sp
import numpy as np
import networkx as nx
import inspect
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def format_data(adj, features, server_hits, labels):
    raw_adj = adj
    adj = nx.adjacency_matrix(nx.from_numpy_matrix(adj))
    print('edges.np', adj.sum() / 2)

    adj_orig, train_adj, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false, client_list, train_label_index = build_test_edges(labels)

    adj_orig.eliminate_zeros()

    features = np.array([preprocess_features(feature) for feature in features])

    raw_distributions = build_real_distributions(server_hits, raw_adj, labels)
    distributions = preprocess_features(raw_distributions)

    raw_adj = np.expand_dims(raw_adj, axis=0)
    adj_norm = adj_to_bias(raw_adj, [raw_adj.shape[1]]*raw_adj.shape[0], nhood=1)

    num_nodes = train_adj.shape[0]

    feature_length = features.shape[-1]

    pos_weight = float(train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) / train_adj.sum()
    norm = train_adj.shape[0] * train_adj.shape[0] / float((train_adj.shape[0] * train_adj.shape[0] - train_adj.sum()) * 2)

    adj_label = train_adj + sp.eye(train_adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    items = [adj, num_nodes, pos_weight, norm, adj_norm, adj_label, features, feature_length, train_adj, train_edges, train_edges_false,
             val_edges, val_edges_false, test_edges, test_edges_false, adj_orig, client_list, raw_distributions, distributions, train_label_index]
    feas = {}
    for item in items:
        feas[retrieve_name(item)] = item

    return feas


def get_placeholder(adj_norm, adj, features):
    placeholders = {
        'features': tf.placeholder(tf.float32, shape=features.shape, name='features'),
        'adj': tf.placeholder(tf.float32, shape=adj_norm.shape, name='adj'),
        'adj_orig': tf.sparse_placeholder(tf.float32, name='adj_orig'),
        'dropout': tf.placeholder(tf.float32, shape=(), name='dropout'),
        'real_distribution': tf.placeholder(dtype=tf.float32, shape=[adj.shape[-1], FLAGS.hidden2],
                                            name='real_distribution')
    }
    return placeholders


def get_model(model_name, placeholders, feature_length, client_list):
    discriminator = Discriminator()
    d_real = discriminator.construct(placeholders['real_distribution'])
    assert model_name == 'GALG' or model_name == 'VGALG'
    if model_name == 'GALG':
        model = GALG(placeholders, feature_length, client_list)
    else:
        model = VGALG(placeholders, feature_length, client_list)
    return d_real, discriminator, model


def get_optimizer(model_name, model, discriminator, placeholders, pos_weight, norm, d_real, num_nodes):
    assert model_name == 'GALG' or model_name == 'VGALG'
    if model_name == 'GALG':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          d_real=d_real,
                          d_fake=discriminator.construct(model.embeddings, reuse=True))
    else:
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model,
                           num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           d_real=d_real,
                           d_fake=discriminator.construct(model.embeddings, reuse=True))
    return opt


def update(model, opt, sess, adj_norm, adj_label, features, placeholders, distributions):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_norm})
    feed_dict.update({placeholders['adj_orig']: adj_label})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    emb = sess.run(model.embeddings, feed_dict=feed_dict)

    z_real_dist = distributions + np.random.randn(distributions.shape[0], distributions.shape[1])
    feed_dict.update({placeholders['real_distribution']: z_real_dist})

    reconstruct_loss = None
    for j in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)

    avg_cost = reconstruct_loss

    return emb, avg_cost


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

    temp_adj = np.zeros((len(client_list), len(client_list)))
    for [i, j] in train_edges:
        temp_adj[i][j] = 1
        temp_adj[j][i] = 1
    for [i, j] in train_edges_false:
        temp_adj[i][j] = 1
        temp_adj[j][i] = 1
    temp_adj = temp_adj.reshape(-1)
    train_label_index = np.where(temp_adj)[0].tolist()

    return adj_orig, train_adj, train_edges, train_edges_false, val_edges, val_edges_false, \
           test_edges, test_edges_false, client_list, train_label_index


def build_real_distributions(server_hits, adj, labels):
    client_list = np.where(labels)[0]
    server_list = np.where(labels == 0)[0]
    print('client_list.np', len(client_list))
    print('server_list.np', len(server_list))

    distributions = np.zeros([adj.shape[-1], adj.shape[-1]])
    for client_id, i in enumerate(client_list):
        count = 0
        for j in server_list:
            if i != j and adj[i][j] == 1:
                distributions[i][j] = server_hits[client_id][count]
                count += 1
    distributions = distributions[client_list, :][:, server_list]
    pca = PCA(n_components=FLAGS.hidden2)
    distributions = pca.fit_transform(distributions)
    print('distributions.shape', distributions.shape)
    return distributions


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
    # return adj_normalized
    return sparse_to_tuple(adj_normalized)


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


def raw_normalization(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features


def adj_to_bias(adj, sizes, nhood=1):
    """
     Prepare adjacency matrix by expanding up to a given neighbourhood.
     This will insert loops on every node.
     Finally, the matrix is converted to bias vectors.
     Expected shape: [graph, nodes, nodes]
    """
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features
