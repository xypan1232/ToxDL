__author__ = 'jasper.zuallaert'

import numpy as np
import GO_Graph_Builder as gogb
import tensorflow as tf

from Layers import dynamic_max_pooling_with_overlapping_windows, BidirectionalGRULayer

# Returns a NetworkObject object, containing the neural network created, and the placeholders that come with it
# The network topology can be constructed with a broad variation of parameters, explained below:
# - type: The topology type ('G' = DeepGO, 'D' = dynamic max pooling, 'K' = k-max pooling, 'O' = zero-padding only,
#                            'R' = GRU, 'M' = single max pooling, 'C' = combined D+K, 'P' = domain vector network only)
# - maxLength: the maximum sequence length, which should be the same as the one specified when reading the Datasets in
#              InputManager.py
# - ngramsize
# - filterSizes: list with the filter size of each convolutional layer                      e.g. [9,7,7]
# - filterAmounts: list with the amount of filters of each convolutional layer              e.g. [100,200,300]
# - maxPoolSizes: list with the max pool sizes of each max pooling layer                    e.g. [2,2,1]
#   Note: filterSizes, filterAmounts and maxPoolSizes should all have the same length
# - sizeOfFCLayers: an integer indicating the size of the fully-connected layers at the end e.g. 64
# - n_of_outputs: the amount of classes for which a prediction should be done by the network
# - dynMaxPoolSize: in case of dynamic or k-max pooling, this integer indicates the amount of output 'buckets'
# - term_indices_file: the mapping file between class indices and GO terms (e.g. inputs/mf.indices)
# - ppi_vectors: a boolean, indicating whether or not domain vectors should be included in the network
# - hierarchy: a boolean, indicating whether or not the hierarchy strategy as described in DeepGO should be added
#              at the output layers
# - embeddingType: the type of embedding used, should be one of 'trainable' or 'onehot'
# - embeddingDepth: in case of trainable embeddings, how large the vector should be is indicated by this integer
# - GRU_state_size: in case of a GRU network, what the size of the hidden state should be is indicated by this integer
def buildNetworkTopology(type,
                         maxLength,
                         ngramsize,
                         filterSizes,
                         filterAmounts,
                         maxPoolSizes,
                         sizeOfFCLayers,
                         n_of_outputs,
                         dynMaxPoolSize,
                         term_indices_file,
                         ppi_vectors,
                         hierarchy,
                         embeddingType,
                         embeddingDepth,
                         GRU_state_size):
    maxLength = maxLength - ngramsize + 1
    X_placeholder = tf.placeholder(tf.int32, [None, maxLength],name='X_placeholder')
    vec_placeholder = tf.placeholder(tf.float32, [None, 256],name='vec_placeholder')
    seqlen_ph = tf.placeholder(tf.int32, [None],name='seqlen_placeholder')
    dropout_placeholder = tf.placeholder(tf.float32,name='dropout_placeholder')

    if type == 'G':
        return NetworkObject(
                buildDeepGO(ngramsize, n_of_outputs, term_indices_file, ppi_vectors, hierarchy),
                X_placeholder,
                seqlen_ph,
                vec_placeholder,
                dropout_placeholder
            )
    elif type in 'DKORMC':
        return NetworkObject(
                buildMyNetwork(type, ngramsize, n_of_outputs, term_indices_file, filterSizes, filterAmounts, maxPoolSizes, sizeOfFCLayers,
                   dynMaxPoolSize, dropout_placeholder, ppi_vectors, hierarchy, embeddingDepth, embeddingType, GRU_state_size),
                X_placeholder,
                seqlen_ph,
                vec_placeholder,
                dropout_placeholder
            )
    elif type == 'P':
        return NetworkObject(
                buildPPIOnlyNetwork(n_of_outputs, term_indices_file, sizeOfFCLayers, hierarchy),
                X_placeholder,
                seqlen_ph,
                vec_placeholder,
                dropout_placeholder
            )
    else:
        return AssertionError('Type {} not supported'.format(type))


#######################################################################################################
#######################################################################################################
#######################################################################################################

# Prints the details of the neural network (layers and output shapes), except for the output layers
def printNeuralNet(layers):
    print('Network information:')
    for l in layers:
        try:
            print('{:35s} -> {}'.format(l.name,l.shape))
        except AttributeError:
            pass

#######################################################################################################
#######################################################################################################
#######################################################################################################

# Returns the DeepGO model. Explanation of the parameters can be found on top of this file
def buildDeepGO(ngramsize, n_of_outputs, term_indices_file, ppi_vectors, hierarchy):
    assert ngramsize == 3
    def network(X, seqlens, vec):
        layers = []
        model = np.random.uniform(-0.05,0.05,(20**ngramsize+1,128))
        model = np.asarray(model,dtype=np.float32)
        model[0][:] = 0.0
        model = tf.Variable(model,trainable=True)
        l = tf.nn.embedding_lookup(model,X,name='embedding_out')
        layers.append(l)
        layers.append(tf.layers.dropout(layers[-1], 0.2))

        layers.append(tf.layers.conv1d(layers[-1],32,128,padding='valid',activation=tf.nn.relu))
        layers.append(tf.layers.max_pooling1d(layers[-1], 64, 32))
        layers.append(tf.contrib.layers.flatten(layers[-1]))

        if ppi_vectors:
            layers.append(tf.concat([layers[-1],vec],axis=1)) ###

        logits = []
        output_layers = [None] * n_of_outputs
        if not hierarchy:
            for i in range(n_of_outputs):
                l1 = tf.layers.dense(layers[-1],256,activation=tf.nn.relu,name='term{}-1'.format(i))
                l2 = tf.layers.dense(l1,1,name='term{}-2'.format(i))
                logits.append(l2)
                output_layers[i] = l2
        else:
            dependencies = gogb.build_graph(term_indices_file)
            fc_layers = {}
            # get all top terms (without parents)
            terms_without_any_more_parents = [term for term in dependencies if not any(1 for parent in dependencies if parent in dependencies[term])]
            # as long as we have more terms without any more parents, loop
            ctr = 0
            while terms_without_any_more_parents:
                ctr+=1
                # create fully-connected layer using the layers[-1] and FC of previous parents
                this_term = terms_without_any_more_parents.pop(0)                                       # get a new term to add
                parents = dependencies[this_term]                                                       # get the parents of this term
                children = list({key for key in dependencies if this_term in dependencies[key]}) # get the children of this term
                prev_l = tf.concat([layers[-1]]+[fc_layers[parent] for parent in parents],axis=1)       # create a FC layer based on the network output + parent fc outputs
                l1 = tf.layers.dense(prev_l, 256, activation=tf.nn.relu,name='term{}-1'.format(this_term))
                fc_layers[this_term] = l1                                                               # add this FC layer to fc_layers
                l2 = tf.layers.dense(l1, 1, name='term{}-2'.format(this_term))                          # create the logit neuron and add to logits and output_layers
                logits.append(l2)
                output_layers[this_term] = l2

                set_of_added_terms = set(terms_without_any_more_parents + list(fc_layers.keys()))       # create a set of all terms that have a FC already
                # check for each child if it is eligible --- i.e. if all of its parents have been covered already
                terms_without_any_more_parents.extend([child for child in children if
                                                                        all(term in set_of_added_terms for term in dependencies[child]) and
                                                                        child not in set_of_added_terms
                                                                        ])

            for term in range(n_of_outputs):
                children_terms = list({key for key in dependencies if term in dependencies[key]})
                if len(children_terms) == 0:
                    output_layers[term] = logits[term]
                else:
                    all_chldrn_l = tf.concat([logits[x] for x in [term]+children_terms],axis=1)
                    mx_l = tf.reduce_max(all_chldrn_l,axis=1,keepdims=True,name='term{}-3'.format(term))
                    output_layers[term] = mx_l

        printNeuralNet(layers)
        print('And then some output layers... ({})\n'.format(len(output_layers)))

        # The output layer here is returned as logits. Sigmoids are added in the TrainingProcedure.py file
        cc = tf.concat(output_layers,axis=1,name='my_logits')
        print('{:35s} -> {}'.format(cc.name, cc.shape))
        return cc

    return network

# Returns one of our models, according to the parameters. Explanation of the parameters can be found on top of this file
def buildMyNetwork(type, ngramsize, n_of_outputs, term_indices_file, filterSizes, filterAmounts, maxPoolSizes, sizeOfFCLayers,
                   dynMaxPoolSize, dropout_placeholder, ppi_vectors, hierarchy, embeddingDepth, embeddingType, GRU_state_size):
    def network(X, seqlens, vec):
        layers = []

        ### Embedding layer ###
        if embeddingType == 'onehot':
            model = np.zeros((20**ngramsize+1,20**ngramsize),dtype=np.float32)
            for i in range(20**ngramsize): model[i+1][i] = 1
        elif embeddingType == 'trainable':
            model = np.random.uniform(-0.05, 0.05, (20 ** ngramsize + 1, embeddingDepth))
            model = np.asarray(model, dtype=np.float32)
            model[0][:] = 0.0
            model = tf.Variable(model, trainable=True)
        else:
            raise AssertionError('embeddingType {embeddingType} unknown')

        l = tf.nn.embedding_lookup(model,X,name='embedding_out')
        layers.append(l)

        ### Convolutional, dropout and maxpool layers ###
        for f_size, f_amount, p_size in zip(filterSizes,filterAmounts,maxPoolSizes):
            layers.append(tf.layers.conv1d(layers[-1],f_amount,f_size,padding='same',activation=tf.nn.relu))
            layers.append(tf.layers.dropout(layers[-1], dropout_placeholder))
            layers.append(tf.layers.max_pooling1d(layers[-1], p_size, p_size))
            seqlens = seqlens // p_size
        print(seqlens)
        ### Varying input strategy, depending on the type ###
        if type == 'D':
            layers.append(dynamic_max_pooling_with_overlapping_windows(layers[-1],seqlens,fixed_output_size=dynMaxPoolSize))
        elif type == 'K':
            layers.append(tf.transpose(layers[-1], perm=[0, 2, 1]))
            values, _indices = tf.nn.top_k(layers[-1], k=dynMaxPoolSize, sorted=False)
            layers.append(values)
        elif type == 'O':
            pass #do nothing special
        elif type == 'R':
            layers.append(BidirectionalGRULayer(layers[-1], seqlens, GRU_state_size))
        elif type == 'M':
            layers.append(tf.layers.max_pooling1d(layers[-1], int(layers[-1].shape[1]), int(layers[-1].shape[1])))
        elif type == 'C':
            D_layer = dynamic_max_pooling_with_overlapping_windows(layers[-1],seqlens,fixed_output_size=dynMaxPoolSize)
            K_layer_pre = tf.transpose(layers[-1], perm=[0, 2, 1])
            K_layer, _indices = tf.nn.top_k(K_layer_pre, k=dynMaxPoolSize, sorted=False)
            K_layer_T = tf.transpose(K_layer, perm=[0, 2, 1])
            layers.append(D_layer)
            layers.append(K_layer_T)
            layers.append(tf.concat([D_layer,K_layer_T],axis=1))
        layers.append(tf.contrib.layers.flatten(layers[-1]))

        ### Concatenate domain vectors if specified ###
        if ppi_vectors:
            layers.append(tf.concat([layers[-1],vec],axis=1))

        ### Build output layers ###
        logits = []
        output_layers = [None] * n_of_outputs
        if not hierarchy:
            for i in range(n_of_outputs):
                l1 = tf.layers.dense(layers[-1],sizeOfFCLayers,activation=tf.nn.relu,name='term{}-1'.format(i))
                l2 = tf.layers.dense(l1,1,name='term{}-2'.format(i))
                logits.append(l2)
                output_layers[i] = l2
        else:
            dependencies = gogb.build_graph(term_indices_file)
            fc_layers = {}
            # get all top terms (without parents)
            terms_without_any_more_parents = [term for term in dependencies if not any(1 for parent in dependencies if parent in dependencies[term])]
            # as long as we have more terms without any more parents, loop
            ctr = 0
            while terms_without_any_more_parents:
                ctr+=1
                # create fully-connected layer using the layers[-1] and FC of previous parents
                this_term = terms_without_any_more_parents.pop(0)                                       # get a new term to add
                parents = dependencies[this_term]                                                       # get the parents of this term
                children = list({key for key in dependencies if this_term in dependencies[key]})        # get the children of this term
                prev_l = tf.concat([layers[-1]]+[fc_layers[parent] for parent in parents],axis=1)       # create a FC layer based on the network output + parent fc outputs
                l1 = tf.layers.dense(prev_l, sizeOfFCLayers, activation=tf.nn.relu,name='term{}-1'.format(this_term))
                fc_layers[this_term] = l1                                                               # add this FC layer to fc_layers
                l2 = tf.layers.dense(l1, 1, name='term{}-2'.format(this_term))                          # create the logit neuron and add to logits and output_layers
                logits.append(l2)
                output_layers[this_term] = l2

                set_of_added_terms = set(terms_without_any_more_parents + list(fc_layers.keys()))       # create a set of all terms that have a FC already
                # check for each child if it is eligible --- i.e. if all of its parents have been covered already
                terms_without_any_more_parents.extend([child for child in children if
                                                                        all(term in set_of_added_terms for term in dependencies[child]) and
                                                                        child not in set_of_added_terms
                                                                        ])

            for term in range(n_of_outputs):
                children_terms = list({key for key in dependencies if term in dependencies[key]})
                if len(children_terms) == 0:
                    output_layers[term] = logits[term]
                else:
                    all_chldrn_l = tf.concat([logits[x] for x in [term]+children_terms],axis=1)
                    mx_l = tf.reduce_max(all_chldrn_l,axis=1,keepdims=True,name='term{}-3'.format(term))
                    output_layers[term] = mx_l

        printNeuralNet(layers)
        print('And then some output layers... ({})\n'.format(len(output_layers)))

        # The output layer here is returned as logits. Sigmoids are added in the TrainingProcedure.py file
        cc = tf.concat(output_layers,axis=1,name='my_logits')
        print('{:35s} -> {}'.format(cc.name, cc.shape))
        return cc

    return network

def buildPPIOnlyNetwork(n_of_outputs, term_indices_file, sizeOfFCLayers, hierarchy):
    def network(X, seqlens, vec):
        layers = []
        layers.append(tf.layers.dense(vec,sizeOfFCLayers,activation=tf.nn.relu)) ###

        #for n_neurons in sizeOfFCLayers[1:]:
        #    layers.append(tf.layers.dense(layers[-1],n_neurons,activation=tf.nn.relu)) ###
        logits = []
        output_layers = [None] * n_of_outputs

        if not hierarchy:
            for i in range(n_of_outputs):
                l1 = tf.layers.dense(layers[-1],64,activation=tf.nn.relu,name='term{}-1'.format(i))
                l2 = tf.layers.dense(l1,1,name='term{}-2'.format(i))
                logits.append(l2)
                output_layers[i] = l2
        else:
            dependencies = gogb.build_graph(term_indices_file)
            fc_layers = {}
            # get all top terms (without parents)
            terms_without_any_more_parents = [term for term in dependencies if not any(1 for parent in dependencies if parent in dependencies[term])]
            # as long as we have more terms without any more parents, loop
            ctr = 0
            while terms_without_any_more_parents:
                ctr+=1
                # create fully-connected layer using the layers[-1] and FC of previous parents
                this_term = terms_without_any_more_parents.pop(0)                                       # get a new term to add
                parents = dependencies[this_term]                                                       # get the parents of this term
                children = list({key for key in dependencies if this_term in dependencies[key]}) # get the children of this term
                prev_l = tf.concat([layers[-1]]+[fc_layers[parent] for parent in parents],axis=1)       # create a FC layer based on the network output + parent fc outputs
                l1 = tf.layers.dense(prev_l, 64, activation=tf.nn.relu,name='term{}-1'.format(this_term))
                fc_layers[this_term] = l1                                                               # add this FC layer to fc_layers
                l2 = tf.layers.dense(l1, 1, name='term{}-2'.format(this_term))                          # create the logit neuron and add to logits and output_layers
                logits.append(l2)
                output_layers[this_term] = l2

                set_of_added_terms = set(terms_without_any_more_parents + list(fc_layers.keys()))       # create a set of all terms that have a FC already
                # check for each child if it is eligible --- i.e. if all of its parents have been covered already
                terms_without_any_more_parents.extend([child for child in children if
                                                                        all(term in set_of_added_terms for term in dependencies[child]) and
                                                                        child not in set_of_added_terms
                                                                        ])

            for term in range(n_of_outputs):
                children_terms = list({key for key in dependencies if term in dependencies[key]})
                if len(children_terms) == 0:
                    output_layers[term] = logits[term]
                else:
                    all_chldrn_l = tf.concat([logits[x] for x in [term]+children_terms],axis=1)
                    mx_l = tf.reduce_max(all_chldrn_l,axis=1,keepdims=True,name='term{}-4'.format(term))
                    output_layers[term] = mx_l

        printNeuralNet(layers)
        print('And then some output layers... ({})\n'.format(len(output_layers)))

        # The output layer here is returned as logits. Sigmoids are added in the TrainingProcedure.py file
        cc = tf.concat(output_layers,axis=1,name='my_logits')
        print('{:35s} -> {}'.format(cc.name, cc.shape))
        return cc

    return network


# Objects of this class hold a neural network tensor, as well as the placeholders used in that network
class NetworkObject:
    def __init__(self, network, X_placeholder, seqlen_ph, vec_placeholder, dropoutPlaceholder):
        self.network = network
        self.X_placeholder = X_placeholder
        self.seqlen_ph = seqlen_ph
        self.vec_placeholder = vec_placeholder
        self.dropoutPlaceholder = dropoutPlaceholder

    def getNetwork(self):
        return self.network

    def getSeqLenPlaceholder(self):
        return self.seqlen_ph

    def get_X_placeholder(self):
        return self.X_placeholder

    def get_vec_placeholder(self):
        return self.vec_placeholder

    def getDropoutPlaceholder(self):
        return self.dropoutPlaceholder


