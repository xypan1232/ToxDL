__author__ = 'jasper.zuallaert'
import InputManager as im
from NetworkTopologyConstructor import buildNetworkTopology
from TrainingProcedure import TrainingProcedure

# The main script for running experiments. It combines calls to different python files.
# - testParameters: an object of TestLauncher.Parameters, indicating what kind of experiment should be executed
def run(testParameters):
    trainDatasetFile, validDatasetFile, testDatasetFile, hierarchy_file = testParameters.datafiles
    print(trainDatasetFile, validDatasetFile, testDatasetFile)
    ### Read in training, validation and test sets ##
    #train_set,valid_set = im.getSequences(trainDatasetFile,testParameters.ngramsize,testParameters.maxLength,
    #                                      testPartDiv=testParameters.testPartDiv,sets_returned = 2)
    train_set = im.getSequences(trainDatasetFile,testParameters.ngramsize,testParameters.maxLength)
    valid_set = im.getSequences(validDatasetFile,testParameters.ngramsize,testParameters.maxLength)
    test_set = im.getSequences_without_shuffle(testDatasetFile,testParameters.ngramsize,testParameters.maxLength)

    ### Build the topology as described in the input file ###
    nn = buildNetworkTopology(type = testParameters.type,
                              maxLength = testParameters.maxLength,
                              ngramsize = testParameters.ngramsize,
                              filterSizes = testParameters.filterSizes,
                              filterAmounts = testParameters.filterAmounts,
                              maxPoolSizes = testParameters.maxPoolSizes,
                              sizeOfFCLayers = testParameters.sizeOfFCLayers,
                              n_of_outputs = train_set.getClassCounts(),
                              dynMaxPoolSize = testParameters.dynMaxPoolSize,
                              term_indices_file= hierarchy_file,
                              ppi_vectors = testParameters.ppi_vectors,
                              hierarchy = testParameters.hierarchy,
                              embeddingDepth = testParameters.embeddingDepth,
                              embeddingType = testParameters.embeddingType,
                              GRU_state_size= testParameters.GRUSize)


    ### Trains the network (and at the end, stores predictions on the test set) ###
    tp = TrainingProcedure(network_object=nn,
                           train_dataset=train_set,
                           valid_dataset=valid_set,
                           test_dataset=test_set,
                           batch_size=testParameters.batchsize,
                           start_learning_rate=testParameters.start_learning_rate,
                           validationFunction=testParameters.validationFunction,
                           update=testParameters.update,
                           dropoutRate=testParameters.dropout,
                           l1reg=testParameters.l1reg,
                           lossFunction=testParameters.lossFunction)

    sess, auROC, auPRC, Fmax, mcc = tp.trainNetwork(testParameters.epochs)

    ### if called from the SingleTermWorkflow.py, get the full training set (train + valid) and test_set, for
    ### usage with integrated gradients (the full train set is used for the baseline there)
    import sys
    if sys.argv[0] == 'ToxDL.py':
        full_train_set = im.getSequences(trainDatasetFile, testParameters.ngramsize, testParameters.maxLength)
        test_set = im.getSequences(testDatasetFile, testParameters.ngramsize, testParameters.maxLength) # this should not be here, but trying if this fixes a bug
        return sess, full_train_set, test_set, auROC, auPRC, Fmax, mcc
    else:
        sess.close()
        #return auROC, auPRC, Fmax
