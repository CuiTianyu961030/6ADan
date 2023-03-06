from extractor import Extractor
from dataloader import format_data, get_placeholder, get_model, get_optimizer, update
from metrics import Metric
from tracker import Tracker
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

flags = tf.app.flags
FLAGS = flags.FLAGS

model_name = 'GALG'
max_attribute_nb = 8
max_attribute_length = 50
checkpt_file = 'linkage_model/GALG/models/cstnet_galg.ckpt'
data_path = 'linkage_model/GALG/data/cstnet.json'

flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', .5 * 0.01, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('n_heads', 4, 'number of graph attention heads.')
flags.DEFINE_integer('n_layers', 2, 'number of graph attention layers.')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 2000, 'number of iterations.')


if __name__ == '__main__':
    print("开始执行 main.py ...")

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
    placeholders = get_placeholder(feas['adj_norm'], feas['adj_orig'], feas['features'])

    # Construct model
    d_real, discriminator, ae_model = get_model(model_name, placeholders, feas['feature_length'], feas['client_list'])

    # Optimizer
    opt = get_optimizer(model_name, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'])

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    linkgen_ap_history = []
    linkgen_auc_history = []
    tracking_ap_history = []
    tracking_auc_history = []

    # Train model
    for epoch in range(FLAGS.iterations):

        emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['distributions'])

        lg_train = Metric(feas['val_edges'], feas['val_edges_false'])
        auc_curr, ap_curr, _ = lg_train.get_scores(emb, feas)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost), "val_auc=", "{:.5f}".format(auc_curr), "val_ap=", "{:.5f}".format(ap_curr))

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

    print('\033[94mTest max scores of address correlation: auc %.5f ap %.5f\033[0m' % (np.max(linkgen_auc_history), np.max(linkgen_ap_history)))
    print('\033[94mTest max scores of user tracking task: auc %.5f ap %.5f\033[0m' % (np.max(tracking_auc_history), np.max(tracking_ap_history)))

    print('\033[95mFinal address correlation auc: %.5f +/- %.5f\033[0m' %
          ((linkgen_auc_history[-1]+linkgen_auc_history[-3])/2, (linkgen_auc_history[-1]-linkgen_auc_history[-3])/2))
    print('\033[95mFinal address correlation ap: %.5f +/- %.5f\033[0m' %
          ((linkgen_ap_history[-1]+linkgen_ap_history[-3])/2, (linkgen_ap_history[-1]-linkgen_ap_history[-3])/2))
    print('\033[95mFinal user tracking auc: %.5f +/- %.5f\033[0m' %
          ((tracking_auc_history[-1]+tracking_auc_history[-3])/2, (tracking_auc_history[-1]-tracking_auc_history[-3])/2))
    print('\033[95mFinal user tracking ap: %.5f +/- %.5f\033[0m' %
          ((tracking_ap_history[-1]+tracking_ap_history[-3])/2, (tracking_ap_history[-1]-tracking_ap_history[-3])/2))

    saver.save(sess, checkpt_file)

    print("执行完成！ ")
