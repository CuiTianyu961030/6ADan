import networkx as nx
import numpy as np
import tensorflow as tf
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


def get_scores(edges_pos, edges_neg, score_matrix, adj_label):
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(score_matrix[edge[0], edge[1]])
        pos.append(adj_label[edge[0], edge[1]])

    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(score_matrix[edge[0], edge[1]])
        neg.append(adj_label[edge[0], edge[1]])

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    auc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return auc_score, ap_score


def get_user_tracking_scores(edges_pos, edges_neg, score_matrix, adj_label):
    test_edges = np.array(edges_pos)
    test_edges_false = np.array(edges_neg)
    test_nodes = np.unique(np.concatenate((test_edges.reshape(-1), test_edges_false.reshape(-1)), axis=0))

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


def jaccard_coefficient(adj, adj_orig, client_list):
    jc_matrix = np.zeros(adj_orig.shape)
    for u, v, p in nx.jaccard_coefficient(nx.from_scipy_sparse_matrix(adj)):
        if u in client_list and v in client_list:
            u = np.where(client_list == u)[0][0]
            v = np.where(client_list == v)[0][0]
            jc_matrix[u][v] = p
            jc_matrix[v][u] = p

    jc_matrix = jc_matrix / jc_matrix.max()
    return jc_matrix


def preferential_attachment(adj, adj_orig, client_list):
    pa_matrix = np.zeros(adj_orig.shape)
    for u, v, p in nx.preferential_attachment(nx.from_scipy_sparse_matrix(adj)):
        if u in client_list and v in client_list:
            u = np.where(client_list == u)[0][0]
            v = np.where(client_list == v)[0][0]
            pa_matrix[u][v] = p
            pa_matrix[v][u] = p

    pa_matrix = pa_matrix / pa_matrix.max()
    return pa_matrix


def common_neighbors(adj, adj_orig, client_list):
    cn_matrix = np.zeros(adj_orig.shape)
    G = nx.from_scipy_sparse_matrix(adj)
    for i in range(adj.shape[0]):
        G.nodes[i]['community'] = 0
    for u, v, p in nx.cn_soundarajan_hopcroft(G):
        if u in client_list and v in client_list:
            u = np.where(client_list == u)[0][0]
            v = np.where(client_list == v)[0][0]
            cn_matrix[u][v] = p
            cn_matrix[v][u] = p

    cn_matrix = cn_matrix / cn_matrix.max()
    return cn_matrix


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

        jc_matrix = jaccard_coefficient(feas['adj'], feas['adj_orig'], feas['client_list'])
        jc_auc, jc_ap = get_scores(feas['test_edges'], feas['test_edges_false'], jc_matrix, feas['adj_orig'])
        print('\033[92mJaccard coefficient correlation scores: auc %.5f ap %.5f\033[0m' % (jc_auc, jc_ap))
        correlation_lg_ap_history.append(jc_ap)
        correlation_lg_auc_history.append(jc_auc)
        jc_auc, jc_ap = get_user_tracking_scores(feas['test_edges'], feas['test_edges_false'], jc_matrix, feas['adj_orig'])
        print('\033[93mJaccard coefficient tracking scores: auc %.5f ap %.5f\033[0m' % (jc_auc, jc_ap))
        tracking_lg_ap_history.append(jc_ap)
        tracking_lg_auc_history.append(jc_auc)

    correlation_lg_ap_history.sort()
    correlation_lg_auc_history.sort()
    tracking_lg_ap_history.sort()
    tracking_lg_auc_history.sort()

    print('\033[96mFinal address correlation auc (LG): %.5f +/- %.5f\033[0m' %
          ((correlation_lg_auc_history[-1] + correlation_lg_auc_history[-3]) / 2, (correlation_lg_auc_history[-1] - correlation_lg_auc_history[-3]) / 2))
    print('\033[96mFinal address correlation ap (LG): %.5f +/- %.5f\033[0m' %
          ((correlation_lg_ap_history[-1] + correlation_lg_ap_history[-3]) / 2, (correlation_lg_ap_history[-1] - correlation_lg_ap_history[-3]) / 2))
    print('\033[96mFinal user tracking auc (LG): %.5f +/- %.5f\033[0m' %
          ((tracking_lg_auc_history[-1] + tracking_lg_auc_history[-3]) / 2, (tracking_lg_auc_history[-1] - tracking_lg_auc_history[-3]) / 2))
    print('\033[96mFinal user tracking ap (LG): %.5f +/- %.5f\033[0m' %
          ((tracking_lg_ap_history[-1] + tracking_lg_ap_history[-3]) / 2, (tracking_lg_ap_history[-1] - tracking_lg_ap_history[-3]) / 2))

    correlation_lg_ap_history = []
    correlation_lg_auc_history = []
    tracking_lg_ap_history = []
    tracking_lg_auc_history = []

    for _ in range(3):

        feas = format_data(net, attributes, distributions, labels)

        pa_matrix = preferential_attachment(feas['adj'], feas['adj_orig'], feas['client_list'])
        pa_auc, pa_ap = get_scores(feas['test_edges'], feas['test_edges_false'], pa_matrix, feas['adj_orig'])
        print('\033[92mPreferential attachment correlation scores: auc %.5f ap %.5f\033[0m' % (pa_auc, pa_ap))
        correlation_lg_ap_history.append(pa_ap)
        correlation_lg_auc_history.append(pa_auc)
        pa_auc, pa_ap = get_user_tracking_scores(feas['test_edges'], feas['test_edges_false'], pa_matrix, feas['adj_orig'])
        print('\033[93mPreferential attachment tracking scores: auc %.5f ap %.5f\033[0m' % (pa_auc, pa_ap))
        tracking_lg_ap_history.append(pa_ap)
        tracking_lg_auc_history.append(pa_auc)

    correlation_lg_ap_history.sort()
    correlation_lg_auc_history.sort()
    tracking_lg_ap_history.sort()
    tracking_lg_auc_history.sort()

    print('\033[96mFinal address correlation auc (LG): %.5f +/- %.5f\033[0m' %
          ((correlation_lg_auc_history[-1] + correlation_lg_auc_history[-3]) / 2, (correlation_lg_auc_history[-1] - correlation_lg_auc_history[-3]) / 2))
    print('\033[96mFinal address correlation ap (LG): %.5f +/- %.5f\033[0m' %
          ((correlation_lg_ap_history[-1] + correlation_lg_ap_history[-3]) / 2, (correlation_lg_ap_history[-1] - correlation_lg_ap_history[-3]) / 2))
    print('\033[96mFinal user tracking auc (LG): %.5f +/- %.5f\033[0m' %
          ((tracking_lg_auc_history[-1] + tracking_lg_auc_history[-3]) / 2, (tracking_lg_auc_history[-1] - tracking_lg_auc_history[-3]) / 2))
    print('\033[96mFinal user tracking ap (LG): %.5f +/- %.5f\033[0m' %
          ((tracking_lg_ap_history[-1] + tracking_lg_ap_history[-3]) / 2, (tracking_lg_ap_history[-1] - tracking_lg_ap_history[-3]) / 2))

    correlation_lg_ap_history = []
    correlation_lg_auc_history = []
    tracking_lg_ap_history = []
    tracking_lg_auc_history = []

    for _ in range(3):

        feas = format_data(net, attributes, distributions, labels)

        cn_matrix = common_neighbors(feas['adj'], feas['adj_orig'], feas['client_list'])
        cn_auc, cn_ap = get_scores(feas['test_edges'], feas['test_edges_false'], cn_matrix, feas['adj_orig'])
        print('\033[92mCommon neighbors correlation scores: auc %.5f ap %.5f\033[0m' % (cn_auc, cn_ap))
        correlation_lg_ap_history.append(cn_ap)
        correlation_lg_auc_history.append(cn_auc)
        cn_auc, cn_ap = get_user_tracking_scores(feas['test_edges'], feas['test_edges_false'], cn_matrix, feas['adj_orig'])
        print('\033[93mCommon neighbors tracking scores: auc %.5f ap %.5f\033[0m' % (cn_auc, cn_ap))
        tracking_lg_ap_history.append(cn_ap)
        tracking_lg_auc_history.append(cn_auc)

    correlation_lg_ap_history.sort()
    correlation_lg_auc_history.sort()
    tracking_lg_ap_history.sort()
    tracking_lg_auc_history.sort()

    print('\033[96mFinal address correlation auc (LG): %.5f +/- %.5f\033[0m' %
          ((correlation_lg_auc_history[-1] + correlation_lg_auc_history[-3]) / 2, (correlation_lg_auc_history[-1] - correlation_lg_auc_history[-3]) / 2))
    print('\033[96mFinal address correlation ap (LG): %.5f +/- %.5f\033[0m' %
          ((correlation_lg_ap_history[-1] + correlation_lg_ap_history[-3]) / 2, (correlation_lg_ap_history[-1] - correlation_lg_ap_history[-3]) / 2))
    print('\033[96mFinal user tracking auc (LG): %.5f +/- %.5f\033[0m' %
          ((tracking_lg_auc_history[-1] + tracking_lg_auc_history[-3]) / 2, (tracking_lg_auc_history[-1] - tracking_lg_auc_history[-3]) / 2))
    print('\033[96mFinal user tracking ap (LG): %.5f +/- %.5f\033[0m' %
          ((tracking_lg_ap_history[-1] + tracking_lg_ap_history[-3]) / 2, (tracking_lg_ap_history[-1] - tracking_lg_ap_history[-3]) / 2))
