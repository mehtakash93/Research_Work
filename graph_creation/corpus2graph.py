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

dataset_folder = "C:\Users\Dakota Medd\Downloads\mfdata"     # For Monthly Notes dataset

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def original_generate_token_graph():
    corp = []
    sentences = []      # Initialize an empty list of sentences
    
    input_folders = [ sub_dir for sub_dir in listdir(dataset_folder) if isdir(join(dataset_folder, sub_dir)) ]
    
    for folder in input_folders:
        dir_path = dataset_folder + os.sep + folder + os.sep
        files = [ f for f in listdir(dir_path) if isfile(join(dir_path,f)) ]
        
        for file in files:
            file_path = dir_path + file
            file_name, file_extension = splitext(file_path)
            doc = ""
            
            if file_extension == ".pdf":
                doc = convert_pdf_to_txt(file_path)
            elif file_extension == ".docx":
                doc = convert_docx_to_txt(file_path)
            else:
                continue
                
            if doc != "":
                doc = doc.decode("utf8")
                #doc = words_to_phrases(doc)
                doc = doc.lower()
                doc = doc_to_wordlist(doc,True)
                corp = it.chain(corp,doc)
                #sentences += doc_to_sentences(doc, tokenizer, remove_stopwords=False)
    
    corp = list(corp)
    graph = nx.Graph()
    weights = Counter()
    edges = set()
    window = corp[0:5]
    
    for tup in it.permutations(window,2):
        weights[tup] += 1
    for i in range(3,len(corp)-2):
        for j in range(i-2,i+2):
            weights[(corp[j],corp[i+2])] += 1
            weights[(corp[i+2],corp[j])] += 1
            edges.add((corp[i+2],corp[j]))
            
    for e in edges:
        graph.add_edge(e[0], e[1], {'weight':weights[e]})
    
    print graph
    nx.write_weighted_edgelist(graph, "graph.g")
    print nx.to_numpy_matrix(graph)
    np.savetxt("graph.adj", nx.to_numpy_matrix(graph))
    print "finished"

def generate_token_graph():
    generate_token_group_graph(group="")

def generate_token_group_graph(group="author",output="graph"):
    corp = defaultdict(list)
    sentences = []      # Initialize an empty list of sentences
    
    input_folders = [ sub_dir for sub_dir in listdir(dataset_folder) if isdir(join(dataset_folder, sub_dir)) ]
    
    for folder in input_folders:
        dir_path = dataset_folder + os.sep + folder + os.sep
        files = [ f for f in listdir(dir_path) if isfile(join(dir_path,f)) ]
        
        for file in files:
            file_path = dir_path + file
            file_name, file_extension = splitext(file_path)
            doc = ""
            
            if file_extension == ".pdf":
                doc = convert_pdf_to_txt(file_path)
            elif file_extension == ".docx":
                doc = convert_docx_to_txt(file_path)
            elif file_extension == ".txt":
                with open(file_path) as f:
                    for line in f:
                        doc = doc+" "+line
            else:
                continue
                
            if doc != "":
                # Extract the author name from the filename
                if group is "author":
                    group = file_name.split("_")[0]
                elif group is "document":
                    group = file_name
                else:
                    group = ""
                    
                doc = doc.decode("utf8")
                doc = doc.lower()
                doc = doc_to_wordlist(doc,True)
                corp[group] = it.chain(corp[group],doc)
                #sentences += doc_to_sentences(doc, tokenizer, remove_stopwords=False)
    
    graph = nx.Graph()
    weights = Counter()
    edges = set()
    
    if group is "":
        coll = list(corp[""])
        window = coll[0:5]
        
        for tup in it.permutations(window,2):
            weights[tup] += 1
            weights[(tup[1],tup[0])] += 1
            edges.add(tup)
            
        for i in range(2,len(coll)-2):
            for j in range(i-2,i+2):
                weights[(coll[j],coll[i+2])] += 1
                weights[(coll[i+2],coll[j])] += 1
                edges.add((coll[i+2],coll[j]))
                
        for e in edges:
            graph.add_edge(e[0], e[1], {'weight':weights[e]})
    else:
        for (g,coll) in corp:
            coll = list(coll)
            window = coll[0:5]
            
            for tup in it.permutations(window,2):
                weights[tup] += 1
                weights[(tup[1],tup[0])] += 1
                edges.add(tup)
            
            for t in window:
                if not (t is g):
                    weights[(g,t)] += 1
                    weights[(t,g)] += 1
                    edges.add((g,t))
                
            for i in range(2,len(coll)-2):
                for j in range(i-2,i+2):
                    if not (coll[i+2] is g):
                        weights[(g,coll[i+2])] += 1
                        weights[(coll[i+2],g)] += 1
                        edges.add((g,coll[i+2]))

                    weights[(coll[j],coll[i+2])] += 1
                    weights[(coll[i+2],coll[j])] += 1
                    edges.add((coll[i+2],coll[j]))
                    
            for e in edges:
                graph.add_edge(e[0], e[1], {'weight':weights[e]})

    nx.write_weighted_edgelist(graph, output+".g")
    print nx.to_numpy_matrix(graph)
    np.savetxt(output+".adj", nx.to_numpy_matrix(graph))
    generate_graph_statistics(graph,output)
    print "finished"
    
def generate_token_group_bigraph(group="author",output="graph", threshold=5):
    corp = defaultdict(list)
    sentences = []      # Initialize an empty list of sentences
    
    input_folders = [ sub_dir for sub_dir in listdir(dataset_folder) if isdir(join(dataset_folder, sub_dir)) ]
    
    for folder in input_folders:
        dir_path = dataset_folder + os.sep + folder + os.sep
        files = [ f for f in listdir(dir_path) if isfile(join(dir_path,f)) ]
        
        for file in files:
            file_path = dir_path + file
            file_name, file_extension = splitext(file_path)
            doc = ""
            
            if file_extension == ".pdf":
                doc = convert_pdf_to_txt(file_path)
            elif file_extension == ".docx":
                doc = convert_docx_to_txt(file_path)
            elif file_extension == ".txt":
                with open(file_path) as f:
                    for line in f:
                        doc = doc+" "+line
            else:
                continue
                
            if doc != "":
                # Extract the author name from the filename
                if group is "author":
                    # Before I just consider the author another token.
                    # Now I need to distinguish the author tokens from
                    # the author identifier
                    group = "a_"+file_name.split("_")[0]
                elif group is "document":
                    group = file_name
                else:
                    group = ""
                    
                doc = doc.decode("utf8")
                doc = doc.lower()
                doc = doc_to_wordlist(doc,True)
                corp[group] = it.chain(corp[group],doc)
                #sentences += doc_to_sentences(doc, tokenizer, remove_stopwords=False)
    
    graph = nx.Graph()
    weights = Counter()
    edges = set()
    
    if group is "":
        coll = list(corp[""])
        window = coll[0:5]
        
        for tup in it.permutations(window,2):
            weights[tup] += 1
            weights[(tup[1],tup[0])] += 1
            edges.add(tup)

        for i in range(2,len(coll)-2):
            for j in range(i-2,i+2):
                weights[(coll[j],coll[i+2])] += 1
                weights[(coll[i+2],coll[j])] += 1
                edges.add((coll[i+2],coll[j]))
                
        for e in edges:
            graph.add_edge(e[0], e[1], {'weight':weights[e]})
    else:
        for (g,coll) in corp.iteritems():
            coll = list(coll)
            window = coll[0:5]
            
            """
            This code adds edges from one token
            to every other token. Need to remove
            for tup in it.permutations(window,2):
                weights[tup] += 1
                weights[(tup[1],tup[0])] += 1
                edges.add(tup)
            """
            for t in window:
                if not (t is g):
                    weights[(g,t)] += 1
                    weights[(t,g)] += 1
                    edges.add((g,t))
                
            for i in range(2,len(coll)-2):
                for j in range(i-2,i+2):
                    if not (coll[i+2] is g):
                        weights[(g,coll[i+2])] += 1
                        weights[(coll[i+2],g)] += 1
                        edges.add((g,coll[i+2]))
                    
                    """
                    This code also adds token to token edges
                    weights[(coll[j],coll[i+2])] += 1
                    weights[(coll[i+2],coll[j])] += 1
                    edges.add((coll[i+2],coll[j]))
                    """
            for e in edges:
                # Bimax and biclique detection techniques do not use weights, so just
                # create threshed edges
                if weights[e] > threshold:
                    graph.add_edge(e[0], e[1])

    #nx.write_weighted_edgelist(graph, output+"_b.g")
    #print nx.to_numpy_matrix(graph)
    #np.savetxt(output+"_b.adj", nx.to_numpy_matrix(graph))
    generate_graph_statistics(graph,output+"_b")
    print "finished"

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
    
    train_index = np.log(train_index.reshape((len(train_index),1))+1)
    test_index = np.log(test_index.reshape((len(test_index),1))+1)
    y_train = y_train.reshape((len(y_train),1))
    y_test = y_test.reshape((len(y_test),1))
    
    model = linear_model.LinearRegression()
    model.fit(train_index,y_train)
    return (np.asscalar(model.coef_),np.asscalar(model.score(test_index,y_test)))
    
#generate_token_group_graph(group="author",output="agraph")
generate_token_group_bigraph(group="author",output="agraph")