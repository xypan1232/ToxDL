__author__ = 'jasper.zuallaert'
import os
# hide tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from InputManager import Dataset
import sys
import time
import Evaluation as eval

# Prepares a TrainingProcedure object for training, using a given network_object and a given training set, following
# the given parameters:
# - network_object: a NetworkObject object as returned by the functions in NetworkTopologyConstructor.py
# - train_dataset: an InputManager.Dataset object
# - valid_dataset: an InputManager.Dataset object
# - test_dataset: an InputManager.Dataset object
# - batch_size: integer
# - start_learning_rate: float
# - validationFunction: the metric which will be looked at during validation, to select the optimal model during
#                       training. Should be one of 'loss', 'f1'
# - update: a string indicating the update strategy. Should be one of 'momentum', 'rmsprop', 'adam'
# - dropoutRate: float
# - l1reg: l1reg multiplier, indicating whether or not L1 regularization should be applied on, and what the multiplier
#               should be if > 0:
#               a) the trainable embedding layer if it is specified
#               b) the first convolutional layer if no trainable embedding is used
# - lossFunction: the loss function type; should be one of 'default' (categorical crossentropy), 'weighted', 'focal'
#                 In the case of weighted loss, the inverse class frequency is used as a multiplier (see code below,
#                 still under experimentation)
class TrainingProcedure:
    def __init__(self, network_object, train_dataset, valid_dataset, test_dataset, batch_size, start_learning_rate,
                 validationFunction, update, dropoutRate, l1reg, lossFunction):
        self.validationFunction = validationFunction
        self.nn = network_object.getNetwork()
        self.n_of_output_classes = test_dataset.getClassCounts()
        self.batch_size = batch_size
        self.X_placeholder = network_object.get_X_placeholder()
        self.vec_placeholder = network_object.get_vec_placeholder()
        self.seqlens_ph = network_object.getSeqLenPlaceholder()
        self.dropout_placeholder = network_object.getDropoutPlaceholder()
        self.Y_placeholder = tf.placeholder(tf.float32, [None, self.n_of_output_classes],name='Y_placeholder')
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.dropoutRate = dropoutRate

        self.predictions_logits = self.nn(self.X_placeholder,self.seqlens_ph,self.vec_placeholder)
        self.sigmoid_f = tf.sigmoid(self.predictions_logits)

        if lossFunction == 'default':
            self.loss_f = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.Y_placeholder,logits=self.predictions_logits)
        elif lossFunction == 'weighted':
            class_counts = self.train_dataset.getCountsPerTerm() #** 2 # get counts
            class_counts = np.maximum(class_counts,np.percentile(class_counts,5))           # if classes less frequent than the given number (to avoid zeros and very low counts)
            class_counts = np.max(class_counts) / class_counts   # get the inverse of division by the maximum value
            class_counts = class_counts / np.max(class_counts)   # normalize to [0,1]
            self.loss_f = tf.math.reduce_mean(class_counts*(tf.math.maximum(self.predictions_logits, 0) - self.predictions_logits * self.Y_placeholder + tf.math.log(1 + tf.math.exp(-abs(self.predictions_logits)))))
        elif lossFunction == 'focal':
            from Layers import focal_loss
            self.loss_f = focal_loss(self.predictions_logits, self.Y_placeholder)

        if l1reg:
            regularization_penalty = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l1_regularizer(scale=l1reg, scope=None),
                tf.trainable_variables()[:1]
            )
            print('NOTE - l1 reg only on first trainable layer')
            self.loss_f = self.loss_f + regularization_penalty

        gs = tf.train.get_or_create_global_step()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if update == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=start_learning_rate, momentum=0.9)
            elif update == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=start_learning_rate)
            elif update == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=start_learning_rate)
            else:
                raise Exception('Unknown update strategy declaration: {}'.format(update))
            self.train_op = self.optimizer.minimize(loss=self.loss_f,global_step=gs)
        self.total_parameters = self._print_num_params()

    # Prints the total number of trainable parameters
    # If this number does not exceed 5 million, and we are not running this class from a SingleTermWorkflow.py call,
    # the session (containing the network parameters) will be stored in the parameters/ directory)
    def _print_num_params(self):
        total_parameters = 0
        # iterating over all variables
        for variable in tf.trainable_variables():
            local_parameters = 1
            shape = variable.get_shape()  # getting shape of a variable
            for i in shape:
                local_parameters *= i.value  # mutiplying dimension values
            total_parameters += local_parameters
        print('This network has {} trainable parameters.'.format(total_parameters))
        if total_parameters < 5000000 and sys.argv[0] != 'SingleTermWorkflow.py':
            print('total_parameters < 5000000 => model will be saved')
        return total_parameters

    # This function trains the network specified initially, with the datasets specified initially
    # Parameters:
    # - epochs: the number of epochs that should be trained
    # Note: based on the initially specified validationFunction, the best model will be used for the final predictions
    def trainNetwork(self,epochs):
        predictions_save_dest = 'predictions/test_{}.txt'.format(time.strftime('%y%m%d-%H%M%S'))
        parameters_save_dest = 'parameters/test_{}'.format(time.strftime('%y%m%d-%H%M%S'))
        ### create session ###
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        self.sess = sess

        ### run initialization ###
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #writer = tf.summary.FileWriter("testgraph.log", sess.graph) for tensorboard usage ; not used in my internship

        self._printOutputClasses(self.train_dataset, 'Training')
        self._printOutputClasses(self.valid_dataset, 'Valid')
        self._printOutputClasses(self.test_dataset, 'Test')

        print(' {:^5} | {:^14} | {:^14} | {:^14} | {:^14} | {:^14} | {:^14} | {:^14} | {:^12} | {:^12}'.format('epoch','train loss','valid loss','tr Fmax','va Fmax','te Fmax','te avgPr','te avgSn','total time','train time'))
        print('-{:-^6}+{:-^16}+{:-^16}+{:-^16}+{:-^16}+{:-^16}+{:-^16}+{:-^16}+{:-^12}-{:-^13}-'.format('','','','','','','','','','','',''))

        ### Pre training, output ##
        best_valid_score = 999999 if self.validationFunction == 'loss' or self.validationFunction == 'fpr' else -1

        t1 = time.time()
        tr_loss, tr_Fmax, tr_avgPr, tr_avgSn = self._evaluateSet(-1, self.train_dataset, 512)
        va_loss, va_Fmax, va_avgPr, va_avgSn = self._evaluateSet(-1, self.valid_dataset, 512)
        te_loss, te_Fmax, te_avgPr, te_avgSn = -1, -1, -1, -1#self._evaluateSet(-1, self.test_dataset, 512)

        print(' {:5d} |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {:4.2f}s     |   {:4.2f}s   '.format(0,tr_loss,va_loss,tr_Fmax,va_Fmax,te_Fmax,te_avgPr,te_avgSn,time.time()-t1,0))

        ### train for each epoch ###
        for epoch in range(1,epochs):
            sys.stdout.flush()
            epoch_start_time = time.time()

            epoch_finished = False
            trainstart = time.time()
            ### train for each batch in this epoch ###
            while not epoch_finished:
                #print(lengths_x.shape, 'length')
                batch_x, lengths_x, batch_y, vector_x, epoch_finished = self.train_dataset.next_batch(self.batch_size)
                #print(lengths_x.shape, 'length')
                sess.run(self.train_op, feed_dict={self.X_placeholder: batch_x, self.Y_placeholder: batch_y, self.vec_placeholder:vector_x, self.seqlens_ph:lengths_x, self.dropout_placeholder:self.dropoutRate})
            trainstop = time.time()

            ### !!! for time-saving purposes, I only calculate the validation metrics - the rest is filled in with -1 ###
            tr_loss, tr_Fmax, tr_avgPr, tr_avgSn = -1,-1,-1,-1
            # tr_loss, tr_Fmax, tr_avgPr, tr_avgSn = self.evaluateSet(epoch, self.train_dataset, 1024)
            va_loss, va_Fmax, va_avgPr, va_avgSn = self._evaluateSet(epoch, self.valid_dataset, 1024)
            # te_loss, te_Fmax, te_avgPr, te_avgSn = self.evaluateSet(epoch, self.test_dataset, 1024)
            te_loss, te_Fmax, te_avgPr, te_avgSn = -1,-1,-1,-1

            print_message = ''
            valid_metric_score = va_loss if self.validationFunction == 'loss' else va_Fmax if self.validationFunction == 'f1' else None
            ### if new best validation result - store the parameters + generate predictions on test set ###
            if valid_metric_score != None and self._compareValidMetrics(valid_metric_score, best_valid_score):
                best_valid_score = valid_metric_score
                self._storeNetworkParameters(parameters_save_dest)
                self._writePredictions(predictions_save_dest)
                print_message = '-> New best valid.'

            print(' {:5d} |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {: 2.7f}   |   {:4.2f}s     |   {:4.2f}s   {}'.format(epoch,tr_loss,va_loss,tr_Fmax,va_Fmax,te_Fmax,te_avgPr,te_avgSn,time.time()-epoch_start_time,trainstop-trainstart,print_message))

        print("Finished")
        print('Parameters should\'ve been stored in {}'.format(parameters_save_dest))

        ### Generate predictions to show at the end of the file, using Evaluation.py  ###
        ### This is done based on the file with predictions that was written, so this ###
        ### could also be achieved by running Evaluation.py after this python program ###
        ### is finished.                                                              ###

        auROC, auPRC, Fmax, mcc = eval.run_eval_per_term(predictions_save_dest)
        if self.n_of_output_classes > 1:
            eval.run_eval_per_protein(predictions_save_dest)
        return sess, auROC, auPRC, Fmax, mcc

    # Generate the losses, f1 scores and other metrics for a given dataset
    # Parameters:
    # - epoch: the epoch which we are at; we could use this to only generate metrics every x epochs (time management);
    #          currently though, this is done at all epochs)
    # - dataset: the Dataset for which we should evaluate
    # - batch_size: the batch size we should use
    # - threshold_range: (default 20) the amount of thresholds we should use to calculate precision, sensitivity, fmax,
    #                    auROC and auPRC (thresholds are uniformly distributed)
    def _evaluateSet(self, epoch, dataset: Dataset, batch_size, threshold_range = 20):
        losses = []
        F_per_thr = []
        avgPr_per_thr = []
        avgSn_per_thr = []

        ### go over each batch and store the losses ###
        batches_done = False
        while not batches_done:
            batch_x, lengths_x, batch_y, vector_x, epoch_finished = dataset.next_batch(batch_size)
            #print(lengths_x.shape, 'length')
            loss_batch = self.sess.run(self.loss_f, feed_dict={self.X_placeholder: batch_x, self.Y_placeholder: batch_y,self.vec_placeholder:vector_x,self.seqlens_ph:lengths_x})
            losses.extend([loss_batch] * len(batch_x))
            if epoch_finished:
                batches_done = True

        ### at the desired epochs (currently: all), do the calculations ###
        if epoch >= 0 and epoch % 1 == 0:
            ph_batch_y = tf.placeholder(tf.float32,shape=(None,dataset.getClassCounts()))
            ph_t = tf.placeholder(tf.float32)
            preds = tf.ceil(self.sigmoid_f - ph_t)
            tp_f = tf.reduce_sum((ph_batch_y + preds) // 2,axis=1)
            number_of_pos_f = tf.reduce_sum(ph_batch_y,axis=1)
            predicted_pos_f = tf.reduce_sum(preds,axis=1)

            ### for every threshold, calculate pr, sn, fscore ###
            for t in range(threshold_range):
                threshold = t/threshold_range
                prSum = 0.0
                snSum = 0.0
                n_of_samples_predicted_pos = 0
                batches_done = False
                while not batches_done: # go over all batches
                    batch_x, lengths_x, batch_y, vector_x, epoch_finished = dataset.next_batch(batch_size)
                    tp_res,n_of_pos_res,predicted_pos_res = self.sess.run([tp_f,number_of_pos_f,predicted_pos_f], feed_dict={ph_batch_y:batch_y,ph_t:threshold,self.X_placeholder: batch_x, self.Y_placeholder: batch_y,self.vec_placeholder:vector_x,self.seqlens_ph:lengths_x})

                    for tp,n_pos,pred_pos in zip(tp_res,n_of_pos_res,predicted_pos_res):
                        if tp:
                            n_of_samples_predicted_pos += 1
                            prSum += tp / pred_pos
                            snSum += tp / n_pos

                    if epoch_finished:
                        batches_done = True

                avgPr = prSum / max(1,n_of_samples_predicted_pos) # number of samples with at least 1 positive prediction
                avgSn = snSum / len(dataset)
                avgPr_per_thr.append(avgPr)
                avgSn_per_thr.append(avgSn)
                F_per_thr.append(2*avgPr*avgSn/(avgPr+avgSn) if avgPr+avgSn > 0 else 0.0)
            Fmax_index = np.argmax(F_per_thr)
            return np.average(losses), F_per_thr[Fmax_index], avgPr_per_thr[Fmax_index], avgSn_per_thr[Fmax_index]
        else:
            return np.average(losses), -1, -1, -1


    # If the number of trainable parameters does not exceed 5 million, and we are not running this class from a
    # SingleTermWorkflow.py call, the session (containing the network parameters) will be stored in the parameters/
    # directory)
    def _storeNetworkParameters(self, saveToDir):
        if self.total_parameters < 5000000:
            try:
                saver = tf.train.Saver()
                if not os.path.exists(saveToDir):
                    os.makedirs(saveToDir)
                saver.save(self.sess,saveToDir+'/'+saveToDir[saveToDir.rfind('/')+1:])
            except Exception:
                print('SOMETHING WENT WRONG WITH STORING SHIT JASPER!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print(sys.exc_info())
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            pass

    # Writes predictions to a file, to be evaluated by Evaluation.py afterwards
    def _writePredictions(self, predictions_save_dest):
        a = open(predictions_save_dest,'w')
        batches_done = False
        while not batches_done:
            batch_x, lengths_x, batch_y, vector_x, names, epoch_finished = self.test_dataset.next_batch_without_shuffle(512)
            sigmoids = self.sess.run(self.sigmoid_f, feed_dict={self.X_placeholder: batch_x, self.Y_placeholder: batch_y,self.vec_placeholder:vector_x,self.seqlens_ph:lengths_x})
            for p,c,n in zip(sigmoids,batch_y, names):
                print(','.join([str(x) for x in p]),file=a)
                print(','.join([str(x) for x in c]),file=a)
                print(n,file=a)
            if epoch_finished:
                batches_done = True

    # Prints the information about the dataset in input
    # - dataset: an InputManager.Dataset object
    # - label: either 'Training', 'Valid', 'Test'
    def _printOutputClasses(self, dataset, label):
        print('{label} set:')
        counts = dataset.getClassCounts()
        if counts == 1:
            print('Number of positives: {dataset.getPositiveCount()}')
            print('Number of negatives: {dataset.getNegativeCount()}')
        else:
            print('Number of {} classes: {}'.format(label,counts))
            print('Number of {} samples: {}'.format(label,len(dataset)))

    # Compares two validation metrics. We could be looking for the minimum (loss) or maximum (f1 score)
    def _compareValidMetrics(self, new, old):
        if self.validationFunction == 'loss':
            return new < old
        else:
            return new > old
