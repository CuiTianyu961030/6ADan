from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np


class Tracker(object):
    def __init__(self, test_edges, test_edges_false, train_edges):
        test_edges = np.array(test_edges)
        test_edges_false = np.array(test_edges_false)
        self.test_nodes = np.unique(np.concatenate((test_edges.reshape(-1), test_edges_false.reshape(-1)), axis=0))
        self.train_edges = train_edges

    def user_tracking_scores(self, emb, feas):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        adj_rec = np.dot(emb, emb.T)

        auc_preds = []
        labels = []
        for i in self.test_nodes:
            for j in self.test_nodes:
                if i != j and [i, j] not in self.train_edges:
                    auc_preds.append(sigmoid(adj_rec[i, j]))
                    labels.append(feas['adj_orig'][i, j])

        auc_score = roc_auc_score(labels, auc_preds)
        ap_score = average_precision_score(labels, auc_preds)

        return auc_score, ap_score

