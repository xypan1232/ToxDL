__author__ = 'jasper.zuallaert'

import Main

# Runs a test using a given file with test specifications
# - spec_file : the location of the testfile with test specifications
# - datafiles: it is possible to call this function from within SingleTermWorkflow.py - in that case,
#              the datafiles will be provided in the function call, and not in the spec_file (the latter are ignored)
def runTest(spec_file, datafiles = None):
    testParameters = Parameters(spec_file)
    testParameters.fileName = spec_file

    # If no datafiles are given, they are deducted from the spec_file. Two possibilities:
    # - We deal with a generated dataset, where the dataset name should start with "Generated - "
    # - We deal with the mf, cc or bp dataset, where the dataset name should be 'mf', 'cc' or 'bp'
    
    testParameters.datafiles = datafiles
    sess, train_set, test_set, auROC, auPRC, Fmax, mcc = Main.run(testParameters)
    return sess, train_set, test_set, auROC, auPRC, Fmax, mcc

# Reads all parameters from the test specifications file, and prints them out. The parameters themselves are explained
# in the documentation. All parameters of the returned object of this class can then just be directly accessed
class Parameters:
    # init with config file, or default if not provided
    def __init__(self, file = None):
        if file == None:
            self._setParameters()
        else:
            r = open(file,'r')
            allLines = r.readlines()
            paramMap = {}
            for line in allLines:
                if len(line.strip()) > 0 and not line.startswith('#'):
                    # import the line
                    p1,p2 = line.split('---')
                    paramMap[p1.strip()] = p2.strip()
            self._setParameters(
                paramMap['type']                                                      if 'type' in paramMap else 'G',
                [int(x.strip()) for x in paramMap['filterSizes'][1:-1].split(',')]    if 'filterSizes' in paramMap else [9,7,7],
                [int(x.strip()) for x in paramMap['filterAmounts'][1:-1].split(',')]  if 'filterAmounts' in paramMap else [200,200,200],
                [int(x.strip()) for x in paramMap['maxPoolSizes'][1:-1].split(',')]   if 'maxPoolSizes' in paramMap else [3,3,3],
                int(paramMap['sizeOfFCLayers'])                                       if 'sizeOfFCLayers' in paramMap else 256,
                int(paramMap['batchsize'])                                            if 'batchsize' in paramMap else 64,
                float(paramMap['start_learning_rate'])                                if 'start_learning_rate' in paramMap else 0.001,
                int(paramMap['epochs'])                                               if 'epochs' in paramMap else 30,
                paramMap['validFunction']                                             if 'validFunction' in paramMap else 'loss',
                paramMap['dataset']                                                   if 'dataset' in paramMap else 'mf',
                paramMap['update']                                                    if 'update' in paramMap else 'adam',
                int(paramMap['maxLength'])                                            if 'maxLength' in paramMap else 1002,
                int(paramMap['embeddingDepth'])                                       if 'embeddingDepth' in paramMap else 15,
                paramMap['embeddingType']                                             if 'embeddingType' in paramMap else 'onehot',
                float(paramMap['dropout'])                                            if 'dropout' in paramMap else 0.2,
                int(paramMap['testPartDiv'])                                          if 'testPartDiv' in paramMap else 7,
                int(paramMap['dynMaxPoolSize'])                                       if 'dynMaxPoolSize' in paramMap else 10,
                int(paramMap['ngramsize'])                                            if 'ngramsize' in paramMap else 1,
                (True if paramMap['ppi_vectors'] == 'True' else False)                if 'ppi_vectors' in paramMap else False,
                (True if paramMap['hierarchy'] == 'True' else False)                  if 'hierarchy' in paramMap else False,
                float(paramMap['l1reg'])                                              if 'l1reg' in paramMap else 0.0,
                int(paramMap['GRUSize'])                                              if 'GRUSize' in paramMap else 256,
                paramMap['lossFunction']                                              if 'lossFunction' in paramMap else 'default',

            )



    def _setParameters(self,
                       type,
                       filterSizes,
                       filterAmounts,
                       maxPoolSizes,
                       sizeOfFCLayers,
                       batchsize,
                       start_learning_rate,
                       epochs,
                       validationFunction,
                       dataset,
                       update,
                       maxLength,
                       embeddingDepth,
                       embeddingType,
                       dropout,
                       testPartDiv,
                       dynMaxPoolSize,
                       ngramsize,
                       ppi_vectors,
                       hierarchy,
                       l1reg,
                       GRUSize,
                       lossFunction
                       ):
        self.type = type
        
        self.filterSizes = filterSizes
        self.filterAmounts = filterAmounts
        self.maxPoolSizes = maxPoolSizes
        self.sizeOfFCLayers = sizeOfFCLayers
        self.batchsize = batchsize
        self.start_learning_rate = start_learning_rate
        self.epochs = epochs
        self.validationFunction = validationFunction
        self.dataset = dataset
        self.update = update
        self.maxLength = maxLength
        self.embeddingDepth = embeddingDepth
        self.embeddingType = embeddingType
        self.dropout = dropout
        self.testPartDiv = testPartDiv
        self.dynMaxPoolSize = dynMaxPoolSize
        self.ngramsize = ngramsize
        self.ppi_vectors = ppi_vectors
        self.hierarchy = hierarchy
        self.l1reg = l1reg
        self.GRUSize = GRUSize
        self.lossFunction = lossFunction

        self._printParameters()

    def _printParameters(self):
        ### Print parameters ###
        print('+{:-<83}+'.format(''))
        print('| {:^81} |'.format('Information for all PARAMETERS used to create this testfile'))
        print('+{:-<83}+'.format(''))
        print('| {:39} | {:<39} |'.format('dataset',self.dataset))
        print('| {:39} | {:<39} |'.format('type',self.type))
        print('| {:39} | {:<39} |'.format('filterSizes',str(self.filterSizes)))
        print('| {:39} | {:<39} |'.format('filterAmounts',str(self.filterAmounts)))
        print('| {:39} | {:<39} |'.format('maxPoolSizes',str(self.maxPoolSizes)))
        print('| {:39} | {:<39} |'.format('sizeOfFCLayers',str(self.sizeOfFCLayers)))
        print('+{:-<83}+'.format(''))
        print('| {:39} | {:<39} |'.format('GRUSize',str(self.GRUSize)))
        print('+{:-<83}+'.format(''))
        print('| {:39} | {:<39} |'.format('batchsize',self.batchsize))
        print('| {:39} | {:<39} |'.format('testPartDiv',self.testPartDiv))
        print('| {:39} | {:<39} |'.format('start_learning_rate',self.start_learning_rate))
        print('| {:39} | {:<39} |'.format('epochs',self.epochs))
        print('+{:-<83}+'.format(''))
        print('| {:39} | {:<39} |'.format('validationFunction',self.validationFunction))
        print('| {:39} | {:<39} |'.format('lossFunction',self.lossFunction))
        print('| {:39} | {:<39} |'.format('update',self.update))
        print('| {:39} | {:<39} |'.format('hierarchy',str(self.hierarchy)))
        print('| {:39} | {:<39} |'.format('l1reg',self.l1reg))
        print('+{:-<83}+'.format(''))
        print('| {:39} | {:<39} |'.format('maxLength',self.maxLength))
        print('| {:39} | {:<39} |'.format('ngramsize',self.ngramsize))
        print('| {:39} | {:<39} |'.format('embeddingDepth',self.embeddingDepth))
        print('| {:39} | {:<39} |'.format('embeddingType',self.embeddingType))
        print('| {:39} | {:<39} |'.format('ppi_vectors',str(self.ppi_vectors)))
        print('+{:-<83}+'.format(''))
        print('| {:39} | {:<39} |'.format('fcDropout',self.dropout))
        print('| {:39} | {:<39} |'.format('dynMaxPoolSize',self.dynMaxPoolSize))
        print('+{:-<83}+'.format(''))
        print('')
