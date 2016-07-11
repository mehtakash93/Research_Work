import networkx as nx
import cPickle

corpusIndex2Word = cPickle.load(open('index2word.pkl', 'rW'))
corpusWord2Index = cPickle.load(open('word2index.pkl', 'rW'))
corpusGraph = nx.read_edgelist('corpusEdgelist.graph')

kgGraph = nx.read_edgelist('knowledgeGraphEdgelist.graph')
kgSynset2Idx = cPickle.load(open('WN_synset2idx.pkl', 'rW'))
kgIdx2Synset = cPickle.load(open('WN_idx2synset.pkl', 'rW'))
kgWord2Synset = cPickle.load(open('WN_concept2synset.pkl', 'rW'))
kgSynset2Word = cPickle.load(open('WN_synset2concept.pkl', 'rW'))

graph = nx.compose(corpusGraph,kgGraph)

#print corpusWord2Index

#print kgWord2Synset


for word in corpusWord2Index:
	if word in kgWord2Synset:
		print "Adding edges for ", word, " between ", corpusWord2Index[word], " and ",kgSynset2Idx[kgWord2Synset[word]]
		graph.add_edge(corpusWord2Index[word],kgSynset2Idx[kgWord2Synset[word]])
		
nx.write_edgelist(graph, "mergedGraphs.graph")





