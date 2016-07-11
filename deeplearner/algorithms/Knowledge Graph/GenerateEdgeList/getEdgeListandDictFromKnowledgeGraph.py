import os
import cPickle

import numpy as np
import scipy.sparse as sp
import networkx as nx

# Put the wordnet-mlj data absolute path here
datapath = "./"
assert datapath is not None

if 'data' not in os.listdir('../'):
    os.mkdir('../data')


def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs

#################################################
### Creation of the synset/indices dictionnaries

np.random.seed(753)

synlist = []
rellist = []

for datatyp in ['train', 'valid', 'test']:
    f = open(datapath + 'wordnet-mlj12-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1])
        synlist += [lhs[0], rhs[0]]
        rellist += [rel[0]]

synset = np.sort(list(set(synlist)))
relset = np.sort(list(set(rellist)))

synset2idx = {}
idx2synset = {}

idx = 0
for i in synset:
    synset2idx[i] = 'kg'+str(idx)
    idx2synset['kg'+str(idx)] = i
    idx += 1
nbsyn = idx
print "Number of synsets in the dictionary: ", nbsyn
# add relations at the end of the dictionary
for i in relset:
    synset2idx[i] = 'kg'+str(idx)
    idx2synset['kg'+str(idx)] = i
    idx += 1
nbrel = idx - nbsyn
print "Number of relations in the dictionary: ", nbrel

f = open('WN_synset2idx.pkl', 'w')
g = open('WN_idx2synset.pkl', 'w')
cPickle.dump(synset2idx, f, -1)
cPickle.dump(idx2synset, g, -1)
f.close()
g.close()

####################################################
### Creation of the synset definitions dictionnaries

f = open(datapath + 'wordnet-mlj12-definitions.txt', 'r')
dat = f.readlines()
f.close()

synset2def = {}
synset2concept = {}
concept2synset={}

for i in dat:
    synset, concept, definition = i[:-1].split('\t')
    synset2def.update({synset: definition})
    word= concept.split("_")
    word=" ".join(word[2:-2])
    synset2concept.update({synset: word})
    concept2synset.update({word: synset})

    


f = open('WN_synset2def.pkl', 'w')
g = open('WN_synset2concept.pkl', 'w')
e=open('WN_concept2synset.pkl', 'w')
cPickle.dump(synset2def, f, -1)
cPickle.dump(synset2concept, g, -1)
cPickle.dump(concept2synset, e, -1)
f.close()
g.close()
e.close()



    #################################################
### Creation of the index files for training
graph = nx.Graph()

for datatyp in ['train', 'valid', 'test']:
    f = open(datapath + 'wordnet-mlj12-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()

    # Declare the dataset variables
    inpl = []
    inpr = []
    inpo = []
    # Fill the sparse matrices
    ct = 0
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1])
        inpl.append(synset2idx[lhs[0]])
        inpr.append(synset2idx[rhs[0]])
        inpo.append(synset2idx[rel[0]])
        ct += 1
        print "Adding data h:t", synset2idx[lhs[0]], synset2idx[rhs[0]]
        graph.add_edge(synset2idx[lhs[0]], synset2idx[rhs[0]])
    # Save the datasets
    if 'data' not in os.listdir('../'):
        os.mkdir('../data')
    f = open('WN-%s-lhs.pkl' % datatyp, 'w')
    g = open('WN-%s-rhs.pkl' % datatyp, 'w')
    h = open('WN-%s-rel.pkl' % datatyp, 'w')
    nx.write_edgelist(graph, "knowledgeGraphEdgelist.graph")
    cPickle.dump(inpl, f, -1)
    cPickle.dump(inpr, g, -1)
    cPickle.dump(inpo, h, -1)
    f.close()
    g.close()
    h.close()
