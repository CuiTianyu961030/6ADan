import numpy as np
import networkx as nx
import random
import tensorflow as tf
from gensim.models import Word2Vec
from extractor import Extractor
from dataloader import format_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

data_path = 'data/cstnet.json'
max_attribute_nb = 8
max_attribute_length = 50

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')

P = 1  # Return hyperparameter
Q = 1  # In-out hyperparameter
WINDOW_SIZE = 3  # Context size for optimization
NUM_WALKS = 1  # Number of walks per source
WALK_LENGTH = 10  # Length of walk per source
DIMENSIONS = 32  # Embedding dimension
DIRECTED = False  # Graph directed/undirected
WORKERS = 8  # Num. parallel workers
ITER = 1  # SGD epochs


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length, verbose=True):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        if verbose == True:
            print('Walk iteration:')
        for walk_iter in range(num_walks):
            if verbose == True:
                print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def get_edge_embeddings(edge_list, emb_matrix):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        emb1 = emb_matrix[node1]
        emb2 = emb_matrix[node2]
        edge_emb = np.multiply(emb1, emb2)
        embs.append(edge_emb)
    embs = np.array(embs)
    return embs


def get_scores(adj, node_list, feas):
    g_train = nx.from_scipy_sparse_matrix(adj)

    g_n2v = Graph(g_train, DIRECTED, P, Q)  # create node2vec graph instance
    g_n2v.preprocess_transition_probs()
    walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)

    train_walks = []
    for walk in walks:
        one_walk = []
        for node in walk:
            one_walk.append(str(node))
        train_walks.append(one_walk)

    model = Word2Vec(train_walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)

    emb_mappings = model.wv

    emb_list = []
    for node_index in node_list:
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)

    pos_train_edge_embs = get_edge_embeddings(feas['train_edges'], emb_matrix)
    neg_train_edge_embs = get_edge_embeddings(feas['train_edges_false'], emb_matrix)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])
    train_edge_labels = np.concatenate([np.ones(len(feas['train_edges'])), np.zeros(len(feas['train_edges_false']))])

    pos_test_edge_embs = get_edge_embeddings(feas['test_edges'], emb_matrix)
    neg_test_edge_embs = get_edge_embeddings(feas['test_edges_false'], emb_matrix)
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])
    test_edge_labels = np.concatenate([np.ones(len(feas['test_edges'])), np.zeros(len(feas['test_edges_false']))])

    edge_classifier = LogisticRegression()
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    test_auc = roc_auc_score(test_edge_labels, test_preds)
    test_ap = average_precision_score(test_edge_labels, test_preds)

    return test_auc, test_ap


def get_user_tracking_scores(adj, node_list, feas):
    g_train = nx.from_scipy_sparse_matrix(adj)

    g_n2v = Graph(g_train, DIRECTED, P, Q)  # create node2vec graph instance
    g_n2v.preprocess_transition_probs()
    walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)

    train_walks = []
    for walk in walks:
        one_walk = []
        for node in walk:
            one_walk.append(str(node))
        train_walks.append(one_walk)

    model = Word2Vec(train_walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)

    emb_mappings = model.wv

    emb_list = []
    for node_index in node_list:
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)

    pos_train_edge_embs = get_edge_embeddings(feas['train_edges'], emb_matrix)
    neg_train_edge_embs = get_edge_embeddings(feas['train_edges_false'], emb_matrix)
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])
    train_edge_labels = np.concatenate([np.ones(len(feas['train_edges'])), np.zeros(len(feas['train_edges_false']))])

    test_edges = np.array(feas['test_edges'])
    test_edges_false = np.array(feas['test_edges_false'])
    test_nodes = np.unique(np.concatenate((test_edges.reshape(-1), test_edges_false.reshape(-1)), axis=0))

    test_edges = []
    test_edge_labels = []
    for i in test_nodes:
        for j in test_nodes:
            if i != j and [i, j] not in feas['train_edges'] and [i, j] not in feas['train_edges_false']:
                test_edges.append([i, j])
                test_edge_labels.append(feas['adj_orig'][i, j])
    test_edge_embs = get_edge_embeddings(test_edges, emb_matrix)

    edge_classifier = LogisticRegression()
    edge_classifier.fit(train_edge_embs, train_edge_labels)

    test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    test_auc = roc_auc_score(test_edge_labels, test_preds)
    test_ap = average_precision_score(test_edge_labels, test_preds)

    return test_auc, test_ap


if __name__ == '__main__':

    data_extractor = Extractor(data_path, max_attribute_nb, max_attribute_length)
    data_extractor.load_data()
    net = data_extractor.build_net()
    attributes = data_extractor.build_attribute_vec()
    distributions = data_extractor.build_distributions()
    labels = data_extractor.build_labels()

    correlation_lg_ap_history = []
    correlation_lg_auc_history = []
    tracking_lg_ap_history = []
    tracking_lg_auc_history = []

    for _ in range(3):

        feas = format_data(net, attributes, distributions, labels)

        test_auc, test_ap = get_scores(feas['adj'], feas['client_list'], feas)
        print('\033[92mnode2vec correlation scores (LG): auc %.5f ap %.5f\033[0m' % (test_auc, test_ap))
        correlation_lg_ap_history.append(test_ap)
        correlation_lg_auc_history.append(test_auc)

        test_auc, test_ap = get_user_tracking_scores(feas['adj'], feas['client_list'], feas)
        print('\033[93mnode2vec tracking scores (LG): auc %.5f ap %.5f\033[0m' % (test_auc, test_ap))
        tracking_lg_ap_history.append(test_ap)
        tracking_lg_auc_history.append(test_auc)

    correlation_lg_ap_history.sort()
    correlation_lg_auc_history.sort()
    tracking_lg_ap_history.sort()
    tracking_lg_auc_history.sort()

    print('\033[96mFinal address correlation auc (LG): %.5f +/- %.5f\033[0m' %
          ((correlation_lg_auc_history[-1]+correlation_lg_auc_history[-3])/2, (correlation_lg_auc_history[-1]-correlation_lg_auc_history[-3])/2))
    print('\033[96mFinal address correlation ap (LG): %.5f +/- %.5f\033[0m' %
          ((correlation_lg_ap_history[-1]+correlation_lg_ap_history[-3])/2, (correlation_lg_ap_history[-1]-correlation_lg_ap_history[-3])/2))
    print('\033[96mFinal user tracking auc (LG): %.5f +/- %.5f\033[0m' %
          ((tracking_lg_auc_history[-1]+tracking_lg_auc_history[-3])/2, (tracking_lg_auc_history[-1]-tracking_lg_auc_history[-3])/2))
    print('\033[96mFinal user tracking ap (LG): %.5f +/- %.5f\033[0m' %
          ((tracking_lg_ap_history[-1]+tracking_lg_ap_history[-3])/2, (tracking_lg_ap_history[-1]-tracking_lg_ap_history[-3])/2))
