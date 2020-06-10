__author__ = 'jasper.zuallaert'
from GO_Graph_Builder import get_all_children_for_term
from random import shuffle

GOA_FILE = 'downloaded_datafiles/goa_uniprot_filtered_experimental.gaf'  # 19 february 2019 download from: ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz
                                                                        # and then filtered on evidence codes EXP|IDA|IPI|IMP|IGI|IEP
UNIPROT_FILE = 'downloaded_datafiles/uniprot_filtered.tsv'    # 18 december 2018 download from: https://www.uniprot.org/uniprot/?query=*&format=tab&force=true&columns=id,entry%20name,sequence,go(biological%20process),go(cellular%20component),go(molecular%20function)&fil=reviewed:yes&compress=yes
                                                                        # filtered (on uniprot website) to only include reviewed entries
                                                                        
### The list of GO terms for which we want to make datasets (one dataset (= 3 files) will be made for each GO term here)
GO_TERM_LIST = [
    ('bp', 'GO:0016310'),
    ('bp', 'GO:0006468'),
]
### The split between train and test set
TRAIN_FRACTION = 0.8
### The amount of samples in the negative set
NEGATIVE_SET_SIZE = 30000

### Create the datasets from a given GO term list
def run(GO_TERM_LIST):
    for ctr, (subontology, go_term_to_add) in enumerate(GO_TERM_LIST):
        print('Doing {subontology}: {go_term_to_add}...')
        createDatasets(subontology, go_term_to_add, TRAIN_FRACTION, NEGATIVE_SET_SIZE)

def createDatasets(subontology, go_term_to_add, train_fraction, negative_set_size):
    ### The locations where the dataset files are to be written
    indices_output_file = 'inputs/self_generated/my_{subontology}_{go_term_to_add.replace(":","_")}.indices'
    sequences_output_file = 'inputs/self_generated/my_{subontology}_{go_term_to_add.replace(":","_")}_{"{}"}.dat'

    positive_goa_ids = set() # collection of positive samples

    ### Get all children of the GO term specified - if one of the children is annotated, this term is also implied
    all_candidate_terms = get_all_children_for_term(go_term_to_add)
    all_candidate_terms.add(go_term_to_add)

    ### Read GOA file (to get the GO term annotations)
    for goa_line in open(GOA_FILE):
        goa_line = goa_line.strip().split('\t')
        id,term,evidence_code = goa_line[1],goa_line[4],goa_line[6]
        if term in all_candidate_terms:
            positive_goa_ids.add(id)

    positive_sequences = [] # collection of positive sequences as tuples: (uniprot_id, sequence)
    negative_sequences = [] # collection of negative sequences as tuples: (uniprot_id, sequence)

    ### Read Uniprot file (only to get the sequence - we do not look at GO annotations here because of the lack of evidence codes)
    for line in open(UNIPROT_FILE).readlines()[1:]:
        line = line.strip().split('\t')
        id, uniprot_id, seq = line[0], line[1], line[2]
        ### Here, we currently follow the DeepGO publication by not ignoring sequences with OUBZJX amino acids
        if not ('O' in seq or 'U' in seq or 'B' in seq or 'Z' in seq or 'J' in seq or 'X' in seq):
            if id in positive_goa_ids:
                positive_sequences.append((uniprot_id, seq))
            else:
                negative_sequences.append((uniprot_id, seq))

    shuffle(positive_sequences)
    shuffle(negative_sequences)

    train_positive_sequences = positive_sequences[:int(len(positive_sequences)*train_fraction)]
    test_positive_sequences =  positive_sequences[int(len(positive_sequences)*train_fraction):]

    train_negative_sequences = negative_sequences[:int(negative_set_size*train_fraction)]
    test_negative_sequences =  negative_sequences[int(negative_set_size*train_fraction):negative_set_size]

    ### Write the indices file (which will just contain one term)
    f = open(indices_output_file, 'w')
    f.write('0 - {go_term_to_add}')
    f.close()

    ### helper function for printing
    def print_output(uniprot_id, seq, label, f):
        print('>{uniprot_id}',file=f)
        print(seq,file=f)
        print(label,file=f)
        print(','.join(['0.0'] * 256),file=f) ### we do not consider the ppi vectors here, so a zero-vector is added

    ### Write fasta files
    f = open(sequences_output_file.format('train'), 'w')
    for uniprot_id, seq in train_positive_sequences:
        print_output(uniprot_id,seq,'1',f)
    for uniprot_id, seq in train_negative_sequences:
        print_output(uniprot_id,seq,'0',f)
    f.close()
    f = open(sequences_output_file.format('test'), 'w')
    for uniprot_id, seq in test_positive_sequences:
        print_output(uniprot_id,seq,'1',f)
    for uniprot_id, seq in test_negative_sequences:
        print_output(uniprot_id,seq,'0',f)
    f.close()

    return (sequences_output_file.format('train'),
            sequences_output_file.format('test'),
            indices_output_file)

### If called as a standalone python script (= not from SingleTermWorkflow.py), the GO_TERM_LIST as specified on top
### of this file is iterated over.
import sys
if sys.argv[0] == 'DatasetCreator.py':
    run(GO_TERM_LIST)
