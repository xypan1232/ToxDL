__author__ = 'jasper.zuallaert, Xiaoyong.Pan'

from DatasetCreator import createDatasets
from TestLauncher import runTest
from IntegratedGradientsRunner import runFromSession
from PosSeqFromSaliencyMapFile import selectPosSeqFromFile
from InterProVisualizer import runInterProVisualizer
from SequenceShowerAA import visualizeSaliencyMapFile
import InputManager as im
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pdb
import joblib
import numpy as np


### the created dataset is picked
NETWORK_SETTINGS_FILE = 'TestFiles/000_test.test'


def run_prediciton(testFile, predictions_save_dest = 'dl.score', save = True):
    test_dataset = im.getSequences_without_shuffle(testFile,1,1002,silent=True)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    saver = tf.train.import_meta_graph('parameters/test_200114-153051/test_200114-153051' +'.meta')
    saver.restore(sess, tf.train.latest_checkpoint('parameters/test_200114-153051/'))
    #saver = tf.train.import_meta_graph('/home/zzegs/workspace/dagw/toxicity_DL/rr/BASF_code/parameters/seq_model/test_190705-110204' +'.meta')
    #saver.restore(sess, tf.train.latest_checkpoint('/home/zzegs/workspace/dagw/toxicity_DL/rr/BASF_code/parameters/seq_model/'))
    graph = tf.get_default_graph()
    prediction_logits = graph.get_tensor_by_name("my_logits:0")
    X_placeholder = graph.get_tensor_by_name("X_placeholder:0")
    Y_placeholder = graph.get_tensor_by_name("Y_placeholder:0")
    seqlen_ph = graph.get_tensor_by_name("seqlen_placeholder:0")
    vec_placeholder = graph.get_tensor_by_name("vec_placeholder:0")
    
    sigmoid_f = tf.sigmoid(prediction_logits)
    #sess.run(output_tensor, feed_dict={.....})
    test_label, test_pred = [], []
    if save:
        a = open('dl.score','w')
    batches_done = False
    while not batches_done:
        batch_x, lengths_x, batch_y, vector_x, epoch_finished = test_dataset.next_batch_without_shuffle(512)
        sigmoids = sess.run(sigmoid_f, feed_dict={X_placeholder: batch_x, Y_placeholder: batch_y, vec_placeholder:vector_x, seqlen_ph:lengths_x})
        for p,c in zip(sigmoids,batch_y):
            if save:
                print(','.join([str(x) for x in p]),file=a)
                print(','.join([str(x) for x in c]),file=a)
            #else:
            test_label.append(c[0])
            test_pred.append(p[0])
        if epoch_finished:
            batches_done = True
    sess.close()
    #if not save:
    return np.array(test_label), np.array(test_pred)

   
def run_motif_scan(testFile, saliencyMapFile):
    testset = im.getSequences(testFile,1,1002,silent=True)
    trainset = im.getSequences('/home/zzegs/workspace/dagw/toxicity_DL/data/train_data_file.dat.domain.toxin',1,1002,silent=True)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    saver = tf.train.import_meta_graph('/home/zzegs/workspace/dagw/toxicity_DL/rr/BASF_code/parameters/seq_model/test_190705-110204' +'.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/home/zzegs/workspace/dagw/toxicity_DL/rr/BASF_code/parameters/seq_model/'))
    runFromSession(0,sess,trainset,testset,useRef=True,outF=saliencyMapFile)
    
    fastaFile, posSaliencyFile = selectPosSeqFromFile(saliencyMapFile)
    visualizeSaliencyMapFile(posSaliencyFile, 'seq_temp')
    sess.close()  
    
def run():
    #the training, validaiton and test set
    datafiles_tuple = ('datasets/train.fa.domain', 'datasets/valid.fa.domain', 'datasets/test.fa.domain', 'toxicity.indices')
    ### train network
    print('>>> TRAINING NETWORK...')
    results = []
    for i in range(10):
        sess, trainset, testset, auROC, auPRC, F1score, MCC = runTest(NETWORK_SETTINGS_FILE, datafiles_tuple)
        sess.close()
        tf.reset_default_graph()
        results.append([auROC, auPRC, F1score, MCC])
    print('auROC', 'auPRC', 'F1socre', 'MCC')
    print(results)
    print('Mean results of 10 runnning')
    print(np.mean(results, axis=0))
    ### build saliency map
    #pdb.set_trace()
    
    

if __name__ == "__main__":
    run()
