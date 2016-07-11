# ==============================================================================
# Name: line.py
# Author: Dakota Medd
# Version: 1.0
# Description: This file contains the LINE class, which manages code pertaining
# to the LINE algorithm.
# ==============================================================================

#import sys
#import math
#sys.path.append("../")
#import numpy as np
#import numpy.random as npr

from subprocess import call
import sys



def createLineEmbedding(filename, line_code="."):
    call([line_code+'/reconstruct', '-train', filename,'-output', 'dense_train.txt', '-depth', '2', '-k-max', '1000'])
    call([line_code+'/line', '-train','dense_train.txt', '-output', 'vec_1st_wo_norm.txt', '-binary', '1', '-size 128', '-order', '1' ,'-negative', '5', '-samples', '1000', '-threads', '40'])
    call([line_code+'/line', '-train','dense_train.txt', '-output', 'vec_2nd_wo_norm.txt', '-binary', '1', '-size 128', '-order', '2' ,'-negative', '5', '-samples', '1000', '-threads', '40'])
    call([line_code+'/normalize', '-input', 'vec_1st_wo_norm.txt', '-output', 'vec_1st.txt', '-binary', '1'])
    call([line_code+'/normalize', '-input', 'vec_2nd_wo_norm.txt', '-output', 'vec_2nd.txt', '-binary', '1'])
    call([line_code+'/concatenate', '-input1', 'vec_1st.txt', '-input2', 'vec_2nd.txt', '-output', 'vec_all.txt', '-binary', '1'])


def main(argv):
    createLineEmbedding(argv[1])


if __name__ == "__main__":
    main(sys.argv)