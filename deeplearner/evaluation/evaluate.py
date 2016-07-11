import sys
sys.path.append("../../")
import numpy as np
import func.tensorflow_io as ti
import sklearn.metrics.pairwise as smp
import matplotlib.pyplot as plt

defaultembed = ti.load_pdlinefile("vec_all.txt")
implementembed = np.load("./line_embedding.npy")
network = ti.read_nxgraph_from_edgelist("facebook_combined.txt")

def create_histogram(embedding):
    cosine = 1-smp.pairwise_distances(embedding, metric="cosine")
    cosine = cosine*100
    cosine = cosine.astype("int")
    
    prob, bins = np.histogram(cosine, bins=16, density=False)
    print len(prob)
    print len(bins)
    plt.plot(bins[1:], prob)

def create_histogram_popup(embedding):
    create_histogram(embedding)
    plt.show()

def create_histogram_file(embedding,outputfile):
    create_histogram(embedding)
    plt.savefig(outputfile)

create_histogram(defaultembed)
create_histogram_file(implementembed,"myhisto.png")