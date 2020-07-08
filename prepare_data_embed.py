import sys
import pdb
import gzip
import re
import numpy as np

TOXIN_GO = 'GO:0090729'
VALID_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I','L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
INVALID_ACIDS = ['U', 'O', 'B', 'Z', 'J', 'X']

def read_fasta_file_new(fasta_file = '../data/all_toxin.gz'):
    seq_dict = {}
    fp = gzip.open(fasta_file, 'r')
    name = ''
    #pdb.set_trace()
    for line in fp:
        #let's discard the newline at the end (if any)
        line = line.decode().rstrip()
        if not len(line):
            continue
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[1:].split('|')[2].split()[0]

            seq_dict[name] = ''
        else:
            #it is sequence
            if not len(name):
                print(line)
            #pdb.set_trace()
            line = re.sub('|'.join(INVALID_ACIDS), '', line)
            seq_dict[name] = seq_dict[name] + line
    fp.close()

    return seq_dict

def read_domain_embedding(embedding_file = 'domain_embedding.gz'):
    domain_embedding = {}
    with gzip.open(embedding_file, 'r') as fp:
         for line in fp:
             values = line.decode().rstrip().split(',')
             domain_embedding[values[0]] = values[1:]
    return domain_embedding


def read_uniprot_id(datafile = 'protein_domain.gz'):
    pro_domain = {}
    with gzip.open(datafile) as fp:
        for line in fp:
            values = line.decode().rstrip().split()
            pro_domain[values[0]] = values[1:]
    return pro_domain


def get_data_from_fasta(fasta_file, embedding_dict, pro_domain):
    fw = open(fasta_file + '.domain', 'w')
    labels = []
    with open(fasta_file, 'r') as fp:
        for line in fp:
            line = line.rstrip()
            if line[0] == '>':
                names = line[1:].split('\t')
                pro_name = names[0]
                label = names[1]
                labels.append(label)
            else:
                seq = line
                fw.write('>' + pro_name + '\n')
                fw.write(seq + '\n')
                fw.write(label + '\n')
                if pro_name in pro_domain:
                    domains = pro_domain[pro_name]
                    tmp = []
                    for dom in domains:
                        if dom not in embedding_dict:
                            continue
                        domain_values = [float(val) for val in embedding_dict[dom]]
                        tmp.append(domain_values)
                    mean_domain = [val for val in np.mean(tmp, axis = 0)]
                    fw.write(','.join(map(str, mean_domain)) + '\n')
                else:
                    fw.write(','.join(['0.0'] * 256)+ '\n')

    fw.close()

fasta_file = sys.argv[1]
embedding_dict = read_domain_embedding()
pro_domain = read_uniprot_id()
get_data_from_fasta(fasta_file, embedding_dict, pro_domain)


