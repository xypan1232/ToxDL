__author__ = 'jasper.zuallaert, Xiaoyong.Pan'
import math
from copy import deepcopy
import numpy as np

# Initialize a dictionary to help with getting amino acid ids from sequence
s1 = 'ACDEFGHIKLMNPQRSTVWY'
d_acids = {c:s1.index(c) for c in s1}

# Returns the id for the amino acid ngram supplied
# +1 at the end is because of the zeropadding (0 = empty amino acid)
def getAminoAcidId(ngram):
    num = 0
    for i in range(len(ngram)):
        num += 20**i * d_acids[ngram[-(i+1)]]
    return num+1

# Reads an input file and returns one or two Dataset objects created from that file
# The input file should consist of 4 lines for each sample:
# - a fasta label               >seq0
# - a protein sequence          MLKIAIRLCAA
# - the class labels            00111010100011
# - a ppi knowledge embedding   0.349,-0.914,0.398,...
# The inputs for this function are as follows:
# - datafile: the location of the file to be read
# - ngramsize: the ngram size that should be used (overlapping n-grams, e.g. for ngramsize = 3, the sequence
#              ACDEF will be cut into ACD, CDE, DEF
# - maxLength: the limit for the sequence length. Shorter lengths are zero-padded until this length is reached,
#              longer sequences are truncated at the end
# - testPartDiv: if this datafile is to be divided into two sets (e.g. training and validation), what should be the
#                distribution? An example: if testPartDiv = 5, the first set will have 4/5 of the samples, and the
#                second set will have 1/5. If testPartDiv = 7, the first set will have 6/7 of the samples, and the
#                second set will have 1/7
# - sets_returned: if this datafile is to be split up in two sets, this can be specified with this parameter
# - silent: if silent == False, it prints out which file this function is reading from
def getSequences(datafile, ngramsize, maxLength, testPartDiv = 0, sets_returned = 1, silent=False):
    maxLength = maxLength - ngramsize + 1
    if not silent:
        print('Reading {}'.format(datafile))
    f = open(datafile)
    allLines = f.readlines()
    names = ['']*(len(allLines)//4) 
    x_data = np.zeros((len(allLines)//4,maxLength),np.int32)
    x_lengths = np.zeros((len(allLines)),np.int32)
    y_data = np.zeros((len(allLines)//4,len(allLines[2].strip())),np.int32)
    vectorData = np.zeros((len(allLines)//4,256))
    for i in range(0,len(allLines),4):
        seq = allLines[i+1].strip()#.replace('.', '')
        #print(seq)
        names[i//4] = allLines[i][1:].strip() 
        allNgrams = [seq[k:k+ngramsize] for k in range(len(seq) - ngramsize + 1)]
        seq = [getAminoAcidId(ngram) for ngram in allNgrams]
        classes = [int(c) for c in allLines[i+2].strip()]
        embeddings = [float(c) for c in allLines[i+3].strip().split(',')]

        for j in range(min(maxLength,len(seq))):
            x_data[i//4][j] = seq[j]
        x_lengths[i//4] = len(seq)
        y_data[i//4] = classes
        vectorData[i//4] = embeddings

    ### shuffle all sequences (this is done for proper training / validation separation)
    idx = np.arange(0, len(x_data))
    np.random.shuffle(idx)  # shuffle indexes
    x_data = x_data[idx]
    x_lengths = x_lengths[idx]
    y_data = y_data[idx]
    vectorData = vectorData[idx]
    names = [names[ind] for ind in idx]

    assert sets_returned in (1,2), 'Only 1 or 2 sets should be returned, not {}'.format(sets_returned)
    if sets_returned == 2:
        assert testPartDiv > 1, 'testPartDiv should be given when sets_returned == 2'
        first_part_len = (testPartDiv-1)*len(x_data)//testPartDiv
        x_data1, x_lengths1, y_data1, vectorData1 = x_data[:first_part_len], x_lengths[:first_part_len], y_data[:first_part_len], vectorData[:first_part_len]
        x_data2, x_lengths2, y_data2, vectorData2 = x_data[first_part_len:], x_lengths[first_part_len:], y_data[first_part_len:], vectorData[first_part_len:]
        return Dataset(x_data1, x_lengths1, y_data1, vectorData1), Dataset(x_data2, x_lengths2, y_data2, vectorData2)
    else:
        return Dataset(x_data,x_lengths,y_data,vectorData, names)

def getSequences_without_shuffle(datafile, ngramsize, maxLength, testPartDiv = 0, sets_returned = 1, silent=False):
    maxLength = maxLength - ngramsize + 1
    if not silent:
        print('Reading {}'.format(datafile))
    f = open(datafile)
    allLines = f.readlines()
    names = ['']*(len(allLines)//4)
    x_data = np.zeros((len(allLines)//4,maxLength),np.int32)
    x_lengths = np.zeros((len(allLines)),np.int32)
    y_data = np.zeros((len(allLines)//4,len(allLines[2].strip())),np.int32)
    vectorData = np.zeros((len(allLines)//4,256))
    for i in range(0,len(allLines),4):
        seq = allLines[i+1].strip()#.replace('.', '')
        names[i//4] = allLines[i][1:].strip()
        #print(seq)
        allNgrams = [seq[k:k+ngramsize] for k in range(len(seq) - ngramsize + 1)]
        seq = [getAminoAcidId(ngram) for ngram in allNgrams]
        classes = [int(c) for c in allLines[i+2].strip()]
        embeddings = [float(c) for c in allLines[i+3].strip().split(',')]

        for j in range(min(maxLength,len(seq))):
            x_data[i//4][j] = seq[j]
        x_lengths[i//4] = len(seq)
        y_data[i//4] = classes
        vectorData[i//4] = embeddings

    assert sets_returned in (1,2), 'Only 1 or 2 sets should be returned, not {}'.format(sets_returned)
    if sets_returned == 2:
        assert testPartDiv > 1, 'testPartDiv should be given when sets_returned == 2'
        first_part_len = (testPartDiv-1)*len(x_data)//testPartDiv
        x_data1, x_lengths1, y_data1, vectorData1 = x_data[:first_part_len], x_lengths[:first_part_len], y_data[:first_part_len], vectorData[:first_part_len]
        x_data2, x_lengths2, y_data2, vectorData2 = x_data[first_part_len:], x_lengths[first_part_len:], y_data[first_part_len:], vectorData[first_part_len:]
        return Dataset(x_data1, x_lengths1, y_data1, vectorData1), Dataset(x_data2, x_lengths2, y_data2, vectorData2)
    else:
        return Dataset(x_data,x_lengths,y_data,vectorData, names)
    
# Class representing a training set, validation set or test set
# It contains
# - sequences (x_data)
# - the sequence lengths (x_lengths)
# - the labels (y_data)
# - the ppi embeddings if supplied, else zerovectors (vector_data)
# The format for the inputs are as follows:
# - x_data: numpy.int32 array of shape (n, seqlen) with n the number of samples and seqlen the maximum sequence length
#           as specified in getSequences(...). Longer sequences will be truncated, shorter sequences are zero-padded.
#           The integers in the sequence indicate the amino acids present (1 = 'A', 2 = 'C', 3 = 'D', etc for unigrams)
# - x_lengths: numpy.int32 array of shape (n) with n the number of samples.
#              The integers indicate the length of each sequence in x_data (the actual length, not the truncated length)
# - y_data: numpy.int32 array of shape (n, n_of_classes) with n the number of samples and n_of_classes the number of
#           annotated GO terms (= classes) in this dataset
# - vector_data: numpy.float32 array of shape (n, 256), containing the PPI vectors as read from the datafiles. For now,
#                the vectors attached with the DeepGO paper are used, and thus the size is fixed to 256
class Dataset:
    def __init__(self, x_data, x_lengths, y_data, vector_data, names = None):
        self.index_in_epoch = 0
        self.test_x_samples = deepcopy(np.concatenate((x_data[:5],x_data[-5:])))
        self.test_y_samples = deepcopy(np.concatenate((y_data[:5],y_data[-5:])))
        self.x_data = x_data
        self.x_lengths = x_lengths
        self.y_data = y_data
        self.num_samples = x_data.shape[0]
        self.vector_data = vector_data
        self.names = names
    # Returns the number of samples in this dataset
    def __len__(self):
        return self.num_samples

    # Returns the maximum sequence length in this dataset
    def getSequenceLength(self):
        return len(self.x_data[0])

    # Returns the number of classes in this dataset
    def getClassCounts(self):
        return len(self.y_data[0])

    # Returns the x_data, x_lengths, y_data and vector_data, but only for the samples in the next batch. It also returns
    # a boolean indicating whether the batch returned is the last batch in the dataset (if so, the next call to
    # next_batch will return the first batch of the next epoch)
    def next_batch(self,batch_size):
        start = self.index_in_epoch
        end = self.index_in_epoch + batch_size

        if start == 0:
            idx = np.arange(0, self.num_samples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexes
            self.x_data = self.x_data[idx]
            self.x_lengths = self.x_lengths[idx]
            self.y_data = self.y_data[idx]
            self.vector_data = self.vector_data[idx]

        if end < self.num_samples:
            self.index_in_epoch = end
            return self.x_data[start:end], self.x_lengths[start:end], self.y_data[start:end], self.vector_data[start:end], False # epoch finished = False
        else:
            self.index_in_epoch = 0
            return self.x_data[start:], self.x_lengths[start:], self.y_data[start:], self.vector_data[start:end], True #epoch finished = True

    def next_batch_without_shuffle(self,batch_size):
        start = self.index_in_epoch
        end = self.index_in_epoch + batch_size

        if start == 0:
            idx = np.arange(0, self.num_samples)  # get all possible indexes
            self.x_data = self.x_data[idx]
            self.x_lengths = self.x_lengths[idx]
            self.y_data = self.y_data[idx]
            self.vector_data = self.vector_data[idx]

        if end < self.num_samples:
            self.index_in_epoch = end
            return self.x_data[start:end], self.x_lengths[start:end], self.y_data[start:end], self.vector_data[start:end], self.names[start:end], False # epoch finished = False
        else:
            self.index_in_epoch = 0
            return self.x_data[start:], self.x_lengths[start:], self.y_data[start:], self.vector_data[start:end], self.names[start:], True #epoch finished = True
        
    # Return the amount of steps per epoch, given a batch_size
    def stepsInEpoch(self,batch_size):
        return math.ceil(len(self) / batch_size)

    def getX(self):
        return self.x_data
    def getY(self):
        return self.y_data
    def getVector(self):
        return self.vector_data
    def getLengths(self):
        return self.x_lengths

    # Returns the amount of positives for each GO term (= for each class)
    def getCountsPerTerm(self):
        return np.sum(np.transpose(self.y_data),axis=1)

    # ONLY USE WHEN ONLY ONE CLASS PRESENT IN DATASET
    # Returns the amount of positive samples in the dataset
    def getPositiveCount(self):
        assert len(self.y_data[0]) == 1
        return int(np.sum(self.y_data))

    # ONLY USE WHEN ONLY ONE CLASS PRESENT IN DATASET
    # Returns the amount of negative samples in the dataset
    def getNegativeCount(self):
        assert len(self.y_data[0]) == 1
        return int(len(self.y_data) - np.sum(self.y_data))
