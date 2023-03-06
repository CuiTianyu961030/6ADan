from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import ipaddress
import json
import string


class Extractor(object):
    def __init__(self, data_path, max_attribute_nb, max_attribute_length):
        self.source = data_path
        self.raw_data = []
        self.node_dict = {}
        self.max_attribute_nb = max_attribute_nb
        self.max_attribute_length = max_attribute_length

    def load_data(self):
        f = open(self.source, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()

        json_string = ''
        for line in lines:
            if line[:-1] == '{':
                json_string = ''
            json_string += line[:-1]
            if line[:-1] == '}':
                self.raw_data.append(json.loads(json_string))

    def build_net(self):
        client_list = []
        server_list = []
        link_list = []
        count = 0
        for user in self.raw_data:
            for i in range(user['nodes']['node_count']):
                client_ip = user['nodes']['node_' + str(i)]['client_features']['ip']
                self.node_dict[client_ip] = count
                client_list.append(count)
                count += 1
                for j in range(user['nodes']['node_'+str(i)]['connection_features']['node_count']):
                    if 'host' in user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)].keys():
                        host = user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)]['host']
                        if host not in self.node_dict.keys():
                            self.node_dict[host] = count
                            server_list.append(count)
                            count += 1
                            link_list.append([client_list[-1], server_list[-1]])
                        else:
                            link_list.append([client_list[-1], self.node_dict[host]])
                    else:
                        server_ip = user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)]['ip']
                        if server_ip not in self.node_dict.keys():
                            self.node_dict[server_ip] = count
                            server_list.append(count)
                            count += 1
                            link_list.append([client_list[-1], server_list[-1]])
                        else:
                            link_list.append([client_list[-1], self.node_dict[server_ip]])

        net = np.zeros((max(self.node_dict.values()) + 1, max(self.node_dict.values()) + 1))
        for link in link_list:
            net[link[0], link[1]] = 1
            net[link[1], link[0]] = 1
        return net

    def build_attribute_vec(self):
        attributes = []
        count = 0

        for user in self.raw_data:
            for i in range(user['nodes']['node_count']):
                attribute = []

                ip = ipaddress.ip_address(user['nodes']['node_'+str(i)]['client_features']['ip']).exploded
                record_version = user['nodes']['node_'+str(i)]['client_features']['record_version']
                client_version = user['nodes']['node_'+str(i)]['client_features']['client_version']
                ciphersuites = user['nodes']['node_'+str(i)]['client_features']['ciphersuites']
                com_method = user['nodes']['node_'+str(i)]['client_features']['com_method']

                # all attribute == feature + tag
                attribute.append(ip.translate(str.maketrans({_: " {0} ".format(_) for _ in string.punctuation})).split() + ['ip'])
                attribute.append([record_version[:3], record_version[3:]] + ['record_version'])
                attribute.append([client_version[:3], client_version[3:]] + ['client_version'])
                attribute.append(list(ciphersuites) + ['ciphersuites'])
                if str(type(com_method)) == '<class \'str\'>':
                    attribute.append([com_method] + ['com_method'])
                else:
                    attribute.append(com_method + ['com_method'])
                attributes.append(attribute)
                count += 1

                for j in range(user['nodes']['node_'+str(i)]['connection_features']['node_count']):
                    keys = sorted(list(user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)].keys() - ['first_connection', 'stream_count']))
                    if 'host' in keys:
                        host = user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)]['host']
                        if self.node_dict[host] == count:
                            attribute = []
                            for key in keys:
                                fingerprint = user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)][key]
                                if key == 'record_version' or key == 'client_version':
                                    attribute.append([fingerprint[:3], fingerprint[3:]] + [key])
                                elif key == 'ip':
                                    fingerprint = ipaddress.ip_address(fingerprint).exploded
                                    fingerprint = fingerprint.translate(
                                        str.maketrans({_: " {0} ".format(_) for _ in string.punctuation}))
                                    attribute.append(fingerprint.split() + [key])
                                else:
                                    fingerprint = fingerprint.translate(str.maketrans({_: " {0} ".format(_) for _ in string.punctuation}))
                                    attribute.append(fingerprint.split() + [key])
                            attributes.append(attribute)
                            count += 1
                        else:
                            for key in keys:
                                flag = 0
                                fingerprint = user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)][key]
                                if key == 'record_version' or key == 'client_version':
                                    current_feature = [fingerprint[:3], fingerprint[3:]]
                                elif key == 'ip':
                                    fingerprint = ipaddress.ip_address(fingerprint).exploded
                                    current_feature = fingerprint.translate(
                                        str.maketrans({_: " {0} ".format(_) for _ in string.punctuation})).split()
                                else:
                                    current_feature = fingerprint.translate(str.maketrans({_: " {0} ".format(_) for _ in string.punctuation})).split()
                                for feature in attributes[self.node_dict[host]]:
                                    if feature == current_feature:
                                        flag = 1
                                if flag == 0:
                                    attributes[self.node_dict[host]].append(current_feature + [key])
                    else:
                        ip = user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)]['ip']
                        if self.node_dict[ip] == count:
                            attribute = []
                            for key in keys:
                                fingerprint = user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)][key]
                                if key == 'record_version' or key == 'client_version':
                                    attribute.append([fingerprint[:3], fingerprint[3:]] + [key])
                                elif key == 'ip':
                                    fingerprint = ipaddress.ip_address(fingerprint).exploded
                                    fingerprint = fingerprint.translate(
                                        str.maketrans({_: " {0} ".format(_) for _ in string.punctuation}))
                                    attribute.append(fingerprint.split() + [key])
                                else:
                                    fingerprint = fingerprint.translate(
                                        str.maketrans({_: " {0} ".format(_) for _ in string.punctuation}))
                                    attribute.append(fingerprint.split() + [key])
                            attributes.append(attribute)
                            count += 1
                        else:
                            for key in keys:
                                flag = 0
                                fingerprint = user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)][key]
                                if key == 'record_version' or key == 'client_version':
                                    current_feature = [fingerprint[:3], fingerprint[3:]]
                                elif key == 'ip':
                                    fingerprint = ipaddress.ip_address(fingerprint).exploded
                                    current_feature = fingerprint.translate(
                                        str.maketrans({_: " {0} ".format(_) for _ in string.punctuation})).split()
                                else:
                                    current_feature = fingerprint.translate(str.maketrans({_: " {0} ".format(_) for _ in string.punctuation})).split()
                                for feature in attributes[self.node_dict[ip]]:
                                    if feature == current_feature:
                                        flag = 1
                                if flag == 0:
                                    attributes[self.node_dict[ip]].append(current_feature + [key])

        model = self.train_doc2vec(attributes)
        # model = self.load_doc2vec()

        attribute_matrix = np.zeros([len(attributes), self.max_attribute_nb, self.max_attribute_length])

        for i in range(len(attributes)):
            for j in range(len(attributes[i])):
                if j == self.max_attribute_nb: break
                attribute_matrix[i][j] = model.docvecs['doc_'+str(i)+'_'+str(j)]

        return attribute_matrix

    def train_doc2vec(self, attributes, fname='linkage_model/GALG/models/doc2vec_model'):
        documents = []
        for i, node_fingerprints in enumerate(attributes):
            for j, doc in enumerate(node_fingerprints):
                documents.append(TaggedDocument(doc[:-1], ['doc_'+str(i)+'_'+str(j), doc[-1]]))
        model = Doc2Vec(documents, vector_size=self.max_attribute_length, window=5, workers=4)
        model.save(fname)
        return model

    def load_doc2vec(self, fname='linkage_model/GALG/models/doc2vec_model'):
        model = Doc2Vec.load(fname)
        return model

    def build_distributions(self):
        server_hits = []
        for user in self.raw_data:
            for i in range(user['nodes']['node_count']):
                hit = []
                for j in range(user['nodes']['node_'+str(i)]['connection_features']['node_count']):
                    hit.append(user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)]['stream_count'])
                server_hits.append(hit)
        return server_hits

    def build_labels(self):
        node_label = np.zeros((max(self.node_dict.values()) + 1))
        index = 0
        user_id = 1
        read_history = []
        for user in self.raw_data:
            for i in range(user['nodes']['node_count']):
                node_label[index] = user_id
                index += 1
                for j in range(user['nodes']['node_'+str(i)]['connection_features']['node_count']):
                    if 'host' in user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)].keys():
                        host = user['nodes']['node_'+str(i)]['connection_features']['node_'+str(j)]['host']
                        if host not in read_history:
                            index += 1
                        read_history.append(host)
                    else:
                        ip = user['nodes']['node_'+str(i)]['connection_features']['node_' + str(j)]['ip']
                        if ip not in read_history:
                            index += 1
                        read_history.append(ip)
            user_id += 1
        return node_label

    def char2token(self, vocabulary_dict, field_value):
        feature_value = []
        for c in field_value:
            if c not in vocabulary_dict.keys():
                feature_value.append(vocabulary_dict['<UNK>'])
            else:
                feature_value.append(vocabulary_dict[c])
        return feature_value
