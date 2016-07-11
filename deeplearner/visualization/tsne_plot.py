#=========================================
# Name:        tsne_plot.py
# Author:      Dakota Medd
# Description: Contains the functions
# required to take an embedding generated
# by LINE or any other method, then
# the plot into a 2 dimensional space using
# tsne.
#=========================================
import sys
sys.path.append("../../")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import func.tensorflow_io as ti
import sklearn.metrics.pairwise as sp

def create_TSNE_embedding(embedding):
    tsne = TSNE(n_components=2)
    return tsne.fit_transform(embedding)

def create_TSNE_plot(embedding):
    tsne = create_TSNE_embedding(embedding)
    
    min_xaxis = np.min(tsne[:,0])-1
    max_xaxis = np.max(tsne[:,0])+1

    min_yaxis = np.min(tsne[:,1])-1
    max_yaxis = np.max(tsne[:,1])+1

    plt.axis([min_xaxis, max_xaxis, min_yaxis, max_yaxis])
    plt.plot(tsne[:,0],tsne[:,1],'bo')

def create_TSNE_popup(embedding):
    create_TSNE_plot(embedding)
    plt.show()

def create_TSNE_plotfile(embedding, outputfile):
    create_TSNE_plot(embedding)
    plt.savefig(outputfile)

def create_TSNE_popup_from_file(filename):
    embedding = ti.load_nplinefile(filename)
    embedding = 1- np.absolute(sp.pairwise_distances(embedding, metric='cosine'))
    create_TSNE_popup(embedding)

def create_TSNE_plotfile_from_file(filename, outputfile):
    embedding = ti.load_nplinefile(filename)
    create_TSNE_plotfile(embedding, outputfile)

if __name__ == "__main__":
    create_TSNE_plotfile_from_file("../algorithms/vec_all.txt", "line1.png")
    #embedding = ti.load_nplinefile("../../complete.txt")
    #print embedding[1]