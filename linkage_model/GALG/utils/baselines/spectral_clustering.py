import networkx as nx
import numpy as np
import tensorflow as tf
from sklearn.manifold import spectral_embedding
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from extractor import Extractor
from dataloader import format_data

data_path = 'data/cstnet.json'
max_attribute_nb = 8
max_attribute_length = 50

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')


def get_scores(edges_pos, edges_neg, embeddings, adj_label):
    score_matrix = np.dot(embeddings, embeddings.T)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        pos.append(adj_label[edge[0], edge[1]])

    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        neg.append(adj_label[edge[0], edge[1]])

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    auc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return auc_score, ap_score


def get_user_tracking_scores(edges_pos, edges_neg, embeddings, adj_label):
    test_edges = np.array(edges_pos)
    test_edges_false = np.array(edges_neg)
    test_nodes = np.unique(np.concatenate((test_edges.reshape(-1), test_edges_false.reshape(-1)), axis=0))

    score_matrix = np.dot(embeddings, embeddings.T)

    preds = []
    labels = []
    for i in test_nodes:
        for j in test_nodes:
            if i != j:
                preds.append(score_matrix[i, j])
                labels.append(adj_label[i, j])

    auc_score = roc_auc_score(labels, preds)
    ap_score = average_precision_score(labels, preds)

    return auc_score, ap_score


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

        emb = spectral_embedding(feas['adj'], n_components=32)[feas['client_list']]

        sc_auc, sc_ap = get_scores(feas['test_edges'], feas['test_edges_false'], emb, feas['adj_orig'])
        print('\033[92mSpectral Clustering correlation scores (LG): auc %.5f ap %.5f\033[0m' % (sc_auc, sc_ap))
        correlation_lg_ap_history.append(sc_ap)
        correlation_lg_auc_history.append(sc_auc)
        sc_auc, sc_ap = get_user_tracking_scores(feas['test_edges'], feas['test_edges_false'], emb, feas['adj_orig'])
        print('\033[93mSpectral Clustering tracking scores (LG): auc %.5f ap %.5f\033[0m' % (sc_auc, sc_ap))
        tracking_lg_ap_history.append(sc_ap)
        tracking_lg_auc_history.append(sc_auc)

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
