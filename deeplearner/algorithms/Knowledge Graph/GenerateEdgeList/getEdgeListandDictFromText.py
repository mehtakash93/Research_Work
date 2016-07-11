#--------------------------------------------------------
# Code written today is the generate_graph_statistics
# method and generate_distribution_constant
#--------------------------------------------------------

import networkx as nx
import itertools as it
import nltk
import re
import os
import json
import cPickle

import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from collections import defaultdict
from os import listdir
from os.path import isfile, isdir, join, splitext
from func.doc_io import convert_pdf_to_txt
from func.doc_io import convert_docx_to_txt
from func.nlp import doc_to_sentences
from func.nlp import doc_to_wordlist
from func.nlp import words_to_phrases
from sklearn import linear_model

import pickle
import gzip
dataset_folder = "/home/priyaranjan/Desktop/IS/ResearchWorkAndNNs/researchfiles/knowledge_graph/bbc"     # For Monthly Notes dataset

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

def parsereviewfile():
    #Extracts the top 1000 reviews from the 5 categories and creates the dataset for graph enbedding.
    ratings = []

    i=0
    for review in parse("reviews_Video_Games.json.gz"):
        ratings.append(review["reviewText"])
        i=i+1
        if i>1000:
            break

    i=0
    for review in parse("reviews_Tools_and_Home_Improvement.json.gz"):
        ratings.append(review["reviewText"])
        i=i+1
        if i>1000:
            break

    i=0
    for review in parse("reviews_Pet_Supplies.json.gz"):
        ratings.append(review["reviewText"])
        i=i+1
        if i>1000:
            break

    i=0
    for review in parse("reviews_Video_Games.json.gz"):
        ratings.append(review["reviewText"])
        i=i+1
        if i>1000:
            break

    i=0
    for review in parse("reviews_Beauty.json.gz"):
        ratings.append(review["reviewText"])
        i=i+1
        if i>1000:
            break

    
    print len(ratings)
    with open("List_5_Categories_Data.txt", 'w') as f:
        for s in ratings:
            f.write(s + '\n')   

def generate_graph_statistics(graph,output):
    # Generated Graph statistics:
    # Degree distribution constant, edge density, number of edges, number of vertices
    # degree of each node
    stats = {}
    stats["nnodes"] = graph.order()
    stats["nedges"] = graph.number_of_edges()
    
    stats["density"] = 2.0*stats["nedges"]/(stats["nnodes"]**2-stats["nnodes"])
    print stats["density"]
    stats["degrees"] = graph.degree()
    stats["wdegrees"] = graph.degree(weight="weight")
    degree_sum = sum(stats["degrees"].values())
    wdegree_sum = sum(stats["wdegrees"].values())
    stats["degrees_dis"] = map(lambda x: float(x)/degree_sum,stats["degrees"].values())
    stats["degrees_dis"].sort()
    stats["wdegree_dis"] = map(lambda x: float(x)/degree_sum,stats["wdegrees"].values())
    stats["wdegree_dis"].sort()
    stats["degree_dis_c"] = generate_distribution_constant(stats["degrees"].values())
    stats["wdegree_dis_c"] = generate_distribution_constant(stats["wdegrees"].values())
    
    with open(output+".stats","w") as f:
        json.dump(stats, f)
    
def generate_distribution_constant(degree_list):
    # Code adapted from the skilearn tutorial:
    # http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#example-linear-model-plot-ols-py
    y_data = np.array(degree_list)
    train_index = np.arange(start=0,stop=len(y_data),step=2)
    test_index = np.arange(start=1,stop=len(y_data),step=2)
    y_train = np.log(y_data[train_index])
    y_test = np.log(y_data[test_index])
    print "Y Train = ", y_train.shape
    train_index = np.log(train_index.reshape((len(train_index),1))+1)
    print "Train Index = ", train_index.shape
    test_index = np.log(test_index.reshape((len(test_index),1))+1)
    y_train = y_train.reshape((len(y_train),1))
    y_test = y_test.reshape((len(y_test),1))
    
    model = linear_model.LinearRegression()
    model.fit(train_index,y_train)
    return (np.asscalar(model.coef_),np.asscalar(model.score(test_index,y_test)))

def read_graph_from_edgelist(file):
	#Reads the input graph file and returns the graph object
	file=open(file, 'rb')
	graph = nx.read_edgelist(file, create_using =nx.MultiGraph(), data=[('weight',int)])
	return graph

def generate_graph_statistics_from_file(file):
	#This function reads the graph text and gennerate graph statistics, stores in a file "graph_b.data"
	graph = read_graph_from_edgelist(file)
	#print graph.edges()
	generate_graph_statistics(graph,"graph_b")

def generate_token_group_bigraph_from_amazon_imdb_dataset(output="graph", threshold=5): 
	#This function is used to create a graph from the imdb/amazon dataset 
	#and store it a file called edgelist.graph in the folder in which the code is being run.
    
    corp = []
    
    input_folders = [ sub_dir for sub_dir in listdir(dataset_folder) if isdir(join(dataset_folder, sub_dir)) ]
    
    for folder in input_folders:
        dir_path = dataset_folder + os.sep + folder + os.sep
        files = [ f for f in listdir(dir_path) if isfile(join(dir_path,f)) ]
        print "Looking inside: " + dir_path
        
        for file in files:
            file_path = dir_path + file
            file_name, file_extension = splitext(file_path)
            print 'Scanning File: ', file_path
            doc = ""
            
            if file_extension == ".txt":
                with open(file_path) as f:
                    print 'Copying data from: ', file_path
                    for line in f:
                        doc = doc+" "+line
            else:
                continue
                
            if doc != "":
                #doc = doc.decode("utf8")
                doc = doc.lower()
                doc = doc_to_wordlist(doc,True)
                print 'Words retrived: ', doc
                corp = it.chain(corp,doc)
    
    graph = nx.Graph()
    weights = Counter()
    edges = list()
    
    
    coll = list(corp)
    wordList = list(set(coll))
    wordDict = {}
    revWordDict = {}

    for i in range(len(wordList)):
        wordDict['text'+str(i)]=wordList[i];
        revWordDict[wordList[i]]='text'+str(i);
	
    window = coll[0:5]
    
    for tup in it.permutations(window,2):
        weights[tup] += 1
        weights[(tup[1],tup[0])] += 1
        print 'Trying to add edge: ', [revWordDict[tup[0]],revWordDict[tup[1]]]
        edges.append([revWordDict[tup[0]],revWordDict[tup[1]]])

    for i in range(2,len(coll)-2):
        for j in range(i-2,i+2):
            edges.append([revWordDict[coll[i+2]],revWordDict[coll[j]]])
            
    for e in edges:
        print 'Trying to add edge: ', e[0],e[1]
        graph.add_edge(e[0], e[1])
    

    #nx.write_weighted_edgelist(graph, output+"_b.g")
    #print nx.to_numpy_matrix(graph)
    #np.savetxt(output+"_b.adj", nx.to_numpy_matrix(graph))

    nx.write_edgelist(graph, "corpusEdgelist.graph")
    g = open('index2word.pkl', 'w')
    cPickle.dump(wordDict, g, -1)
    g.close()
    g = open('word2index.pkl', 'w')
    cPickle.dump(revWordDict, g, -1)
    g.close()

    print "finished"
    

    
generate_token_group_bigraph_from_amazon_imdb_dataset(output="agraph")
#generate_graph_statistics_from_file("edgelist.graph")

#parsereviewfile()
