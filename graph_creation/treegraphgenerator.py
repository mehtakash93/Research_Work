#==============================================
# Name:        treegraphgenerator.py
# Author:      Dakota Medd
# Description: This module takes
# in an edge list, then randomly induces
# a set of subgraphs, which are then connected
# with a single edge
#==============================================

import sys
sys.path.append("../")

import networkx as nx
import func.tensorflow_io as ti
import random as rd
import deeplearner.algorithms.line_cpp.line_subprocess as line

def sample_subgraphs(graph,num_subgraphs):
    vertices = rd.sample(xrange(graph.vcount()),num_subgraphs)
    for i in vertices:
        neighbors = graph.neighborhood(i,order=3)
    return (len(neighbors), graph.vcount())

def create_tree_structure(files):
    graphs = nx.MultiGraph()
    for f in files:
        graphs = nx.disjoint_union(graphs, ti.read_nxgraph_from_edgelist(f))
    return graphs

if __name__ == "__main__":
    #graph = ti.read_igraph_from_edgelist("../data/facebook/0.edges")
    #s = sample_subgraphs(graph, num_subgraphs=4)
    #print s
    graph = ti.read_nxgraph_from_edgelist("../data/facebook/0.edges")
    nx.write_edgelist(graph, "testcomplete.edges",data=False)
    
    graph.remove_node('203')
    nx.write_edgelist(graph, "testincomplete.edges",data=False)

    line.createLineEmbedding("testcomplete.edges", "../deeplearner/algorithms/line_cpp")
    line.createLineEmbedding("testincomplete.edges", "../deeplearner/algorithms/line_cpp")
    #graph = create_tree_structure(["../data/facebook/0.edges","../data/facebook/107.edges"])#,"../data/facebook/348.edges"])
    #nx.write_edgelist(graph, "test.edges", data=False)
    #g = ti.read_igraph_from_edgelist("test.edges")
    #layout = g.layout("lgl")
    #ig.plot(g, layout = layout)
    #for i in s:
    #    print i