from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np


class Metric(object):
    def __init__(self, edges_pos, edges_neg):
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg

    def get_scores(self, emb, feas):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        adj_rec = np.dot(emb, emb.T)

        preds_pos = []
        pos = []
        for e in self.edges_pos:
            preds_pos.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(feas['adj_orig'][e[0], e[1]])

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(feas['adj_orig'][e[0], e[1]])

        preds_all = np.hstack([preds_pos, preds_neg])
        labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

        auc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return auc_score, ap_score, emb
