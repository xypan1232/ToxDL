__author__ = 'jasper.zuallaert'
import sys
import numpy as np
import InputManager as im
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


MAXIMUM_LENGTH = 1002 # hard-coded maximum length, for now

# Called from SingleTermWorkflow, or as a standalone python script
# Takes a trained model, and generates saliency maps for all sequences in a given test_dataset
# The train_dataset given is used for calculating the reference, if desired
# Note that this should only be used in case of one-hot encoding
# Other inputs are Tensorflow tensors/placeholders and a session object
# - termN: the index of the term in the Y labels (0 if only one term in dataset)
# - prediction_logits: tf Tensor of shape (n, c) with n the number of samples in the batch, and c the number of classes
#                      in the dataset
# - sess: The tf session object containing the model
# - X_ph: Placeholder as initialized in NetworkTopologyConstructor.py
# - seqlens_ph: Placeholder as initialized in NetworkTopologyConstructor.py
# - dropout_ph: Placeholder as initialized in NetworkTopologyConstructor.py
# - train_dataset: Dataset object, which should be the training+validation set (used to calculate the reference)
# - test_dataset: Dataset object, containing the test set (for which saliency maps will be generated)
# - use_reference: Indicates whether to run Integrated Gradients with a reference (= the average amino acid frequencies
#                  or without (=
#                  Because of varying input lengths, the frequencies are calculated separately for the first 'ran'
#                  (variable name, see code; by default 5) and the last 'ran' amino acids, and for all the ones in
#                  between, the average for all amino acids from position 5 until -5 is calculated.
# - outF: An output file to write to (can be None for standard output)
def runIntegratedGradientsOnTestSet(termN,
                                    predictions_logits,
                                    sess,
                                    X_ph,
                                    seqlens_ph,
                                    dropout_ph,
                                    train_dataset,
                                    test_dataset,
                                    use_reference = False,
                                    outF = None):
    graph = tf.get_default_graph()
    ### get the tensor that yields the embedding output (one-hot encoding)
    embedding_f = graph.get_tensor_by_name("embedding_out:0")
    ### get the logit of the one term we're interested in
    term_logit = tf.expand_dims(tf.gather(predictions_logits, termN, axis=1), 1)
    ### tensor for gradient calculation on that embedding output
    gs = tf.gradients(term_logit, embedding_f)
    epoch_finished = False
    outFile = open(outF, 'w') if outF else None

    ### Calculate the reference (ran first positions, ran last positions and an average for all positions in between)
    ran = 5
    freqs = np.zeros((ran * 2 + 1, 20), dtype=np.float32)
    if use_reference:
        for sequence, seqlen in zip(train_dataset.getX(),train_dataset.getLengths()):
            seqlen = min(MAXIMUM_LENGTH, seqlen)
            for pos in range(ran):
                freqs[pos][int(sequence[pos]-1)] += 1
                freqs[-pos-1][int(sequence[seqlen-pos-1]-1)] += 1
            for pos in range(ran,seqlen-ran):
                freqs[ran][int(sequence[pos]-1)] += 1
        for pos in range(ran*2+1):
            freqs[pos] /= sum(freqs[pos])

    ### Increase num_integration_steps for higher precision
    ### Here, for each step, the gradient is calculated for the difference with the reference (which increases with
    ### each step)
    num_integration_steps = 30
    while not epoch_finished:
        batch_x, lengths_x, batch_y, vector_data, epoch_finished = test_dataset.next_batch(1024)
        lengths_x = [min(x,MAXIMUM_LENGTH) for x in lengths_x] # max 1002 by default!
        embedding_results = sess.run(embedding_f, feed_dict={X_ph: batch_x, seqlens_ph: lengths_x})

        ### Calculate the difference from reference
        if use_reference:
            difference_part = np.zeros_like(embedding_results)
            for seq_n in range(len(batch_x)):
                for pos in range(ran):
                    difference_part[seq_n][pos] = (embedding_results[seq_n][pos] - freqs[pos]) / num_integration_steps
                for pos in range(ran,lengths_x[seq_n]-ran):
                    difference_part[seq_n][pos] = (embedding_results[seq_n][pos] - freqs[ran]) / num_integration_steps
                for pos in range(lengths_x[seq_n]-ran,lengths_x[seq_n]):
                    difference_part[seq_n][pos] = (embedding_results[seq_n][pos] - freqs[pos-lengths_x[seq_n]]) / num_integration_steps

            # k = 3 # this was some code to check correctness. Leaving it here for now
            # print(lengths_x[k])
            # for aa_n in range(20):
            #     print(','.join(['{: .3f}'.format(difference_part[k][index][aa_n][0]) for index in range(lengths_x[k])]))
            # print()
            # for aa_n in range(20):
            #     print(','.join(['{:6d}'.format(int(embedding_results[k][index][aa_n][0])) for index in range(lengths_x[k])]))
            # print()
            # for aa_n in range(20):
            #     print(','.join(['{: .3f}'.format(freqs[index][aa_n][0]) for index in range(ran*2+1)]))
            # exit()
        else:
            difference_part = embedding_results / num_integration_steps

        ### Calculate the gradients for each step
        allNucs = batch_x
        allClasses = [y[termN] for y in batch_y]
        allSeqLens = lengths_x
        allValues = np.zeros((len(batch_x), len(batch_x[0]),20), np.float32)
        allPreds = [p[termN] for p in sess.run(tf.math.sigmoid(predictions_logits),feed_dict={X_ph: batch_x, seqlens_ph: lengths_x,dropout_ph: 0.0})]

        for step in range(1, num_integration_steps + 1):
            baseline = np.zeros_like(embedding_results)
            if use_reference:
                for seq_n in range(len(batch_x)):
                    for pos in range(ran):
                        baseline[seq_n][pos] = freqs[pos]
                    for pos in range(ran,lengths_x[seq_n]-ran):
                        baseline[seq_n][pos] = freqs[ran]
                    for pos in range(lengths_x[seq_n]-ran,lengths_x[seq_n]):
                        baseline[seq_n][pos] = freqs[pos-lengths_x[seq_n]]
            batch_x_for_this_step_1 = baseline + difference_part * (step - 1)
            batch_x_for_this_step_2 = baseline + difference_part * step
            all_gradients_1 = sess.run(gs, feed_dict={embedding_f: batch_x_for_this_step_1, seqlens_ph: lengths_x,dropout_ph: 0.0})[0]
            all_gradients_2 = sess.run(gs, feed_dict={embedding_f: batch_x_for_this_step_2, seqlens_ph: lengths_x,dropout_ph: 0.0})[0]

            allValues += (all_gradients_1 + all_gradients_2) / 2 * difference_part

        ### Generate outputs. Note that the sequence printed out could be truncated if the actual length surpasses the
        ### maximum length (1002 by default)
        for pred, seq, cl, seqlen, values in zip(allPreds, allNucs, allClasses, allSeqLens, allValues):
            print('{},{},actual_length={}'.format(pred, cl, seqlen),file=outFile)
            print(','.join(['_ACDEFGHIKLMNPQRSTVWY'[int(nuc)] for nuc in seq[:seqlen]]),file=outFile)
            print(','.join([str(score[int(nuc)-1]) for score, nuc in zip(values[:seqlen], seq[:seqlen])]),file=outFile)

# Function to call if we want to use IntegratedGradients.py from another file (such as SingleTermWorkflow.py)
# - For parameters, see the explanation for the function above
def runFromSession(termN, sess, train_set, test_set, useRef = True, outF = None):
    graph = tf.get_default_graph()
    prediction_logits = graph.get_tensor_by_name("my_logits:0")
    X_placeholder = graph.get_tensor_by_name("X_placeholder:0")
    seqlen_ph = graph.get_tensor_by_name("seqlen_placeholder:0")
    dropout_ph = graph.get_tensor_by_name("dropout_placeholder:0")

    runIntegratedGradientsOnTestSet(termN, prediction_logits, sess, X_placeholder, seqlen_ph, dropout_ph, train_set, test_set, useRef, outF)

# If called as a standalone python script, it should have the 5 arguments as stated below
if len(sys.argv) != 6 and sys.argv[0] == 'IntegratedGradientsRunner.py':
    print('Usage: python IntegratedGradientsRunner.py <term number> <parameter file> <train file> <test file> <use_reference>')
elif sys.argv[0] == 'IntegratedGradientsRunner.py':
    termN = int(sys.argv[1])
    paramFile = sys.argv[2] # e.g. 'parameters/test_181212_225609
    trainFile = sys.argv[3] # e.g. 'inputs/mf_train.dat'
    testFile = sys.argv[4]  # e.g. 'inputs/mf_test.dat'
    useRef = bool(sys.argv[5])

    train_set = im.getSequences(trainFile,1,MAXIMUM_LENGTH,silent=True)
    test_set = im.getSequences(testFile,1,MAXIMUM_LENGTH,silent=True)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    paramFile = paramFile
    paramFileFullName = paramFile + '/' + paramFile[paramFile.rfind('/') + 1:]
    saver = tf.train.import_meta_graph(paramFileFullName + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(paramFile))

    runFromSession(termN, sess, train_set, test_set, useRef=useRef)
