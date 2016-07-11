# ==============================================================================
# Name: dline.py
# Author: Dakota Medd
# Version: 1.0
# Description: This file contains the tensorflow code defining the different
# approaches to incorporating graph structure
# ==============================================================================

import sys
sys.path.append("../")

import tensorflow as tf
import numpy as np
import tensorflow_utils as tu
import tensorflow_io as ti
import networkx as nx
class DLINE(object):
    def __init__(self):
        super(DLINE, self).__init__()
        self.pairs = []

    def load_graph_into_DLINE(self,graph):
        """
            Takes in a networkx Graph object and converts the edges into
            tuples of indices, representing the input
            and expected output used to train LINE

            Parameters:
                graph: A networkx graph object
            Return:
                A list of tuples, denoting edges with the node id being
                their one-hot encoded representation
        """

        # Note to self: Have Master's students generalize to any arbitary structure
        pairs = []

        nodes = graph.nodes()
        structs = []

        for e in nodes:
            structs.append(e+"_egonet")

        index_nodes = tu.build_vocab(nodes)
        index_structs = tu.build_vocab(structs)

        for n in nodes:
            index_in = []
            for e in nx.all_neighbors(graph,n):
                index_in.append(index_nodes[e])

            pairs.append((index_in, [index_structs[n+"_egonet"]]))

        self.pairs = pairs
        return pairs

    def cast_to_input(self, pair, training_placeholder, test_placeholder):
        return {training_placeholder: pair[0], test_placeholder: [pair[1]]}

    def tuple_generator(self,training_placeholder, test_placeholder):
        """
            Generator object that allows the class to feed the input/output pairs
            into the session.run instance inside the tensorflow pipeline
        """
        for i in self.pairs:
            yield self.cast_to_input(i, training_placeholder, test_placeholder)

    def build_DLINE(self, node_size, struct_size, dim_size, num_sampled, train_inputs, train_labels, valid_dataset):
        """
            This method will build a tensorflow computation graph representing one option to learn 
            embeddings. This function uses the default computation graph. It does not create a new
            computation graph, and therefore the default computation graph needs to be created before this
            function is ran inside it's with statement.

            This approach mirrors Doc2Vec's distributed memory approach. Each edge in the structure is encoded
            in a binary representation. The structures are also encoded in a separate binary representation. The
            goal of the algorithm is to learn the weights to embed the distributed representation into the structural
            representation.


            Parameters:
                node_size: The number of nodes inside the graph
                struct_size: The number of structures inside the graph
                dim_size: The number of dimensions for the learned embedding
                num_sampled: The number of edges to sample for negative-sampling (read paper)
                train_inputs: The placeholder tensorflow node that represents the input node
                train_labels: The placeholder tensorflow node that represents the output node
                validation_dataset: The index vectors for node's whose neighor to node similarity
                    you want to check during the validation set
            Return:
                The tensorflow node objects for:
                    the optimizer
                    loss function
                    normalized_embeddings
                    valid_embeddings
                    similarity computation
        """

        distributed_embedding = tu.create_uweight(node_size, dim_size, "disembedding")
        structure_embedding = tu.create_tnweight(struct_size, dim_size, "structembedding")
        bias = tu.create_bias(struct_size, 0, "struct_bias")

        # Look up embeddings for inputs.
        dis_embed = tf.expand_dims(tf.reduce_mean(tf.nn.embedding_lookup(distributed_embedding, train_inputs),0),0)
        struct_embedd = tf.nn.embedding_lookup(structure_embedding, train_labels)

        loss = tf.nn.nce_loss(structure_embedding, bias, dis_embed, train_labels, num_sampled, struct_size)
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        """
        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

        normalized_embeddings = embeddings / norm

        # Get the embedding for the validation dataset
        valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)

        # Compute the full similarity matrix
        similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)
        """
        #return (optimizer,loss,normalized_embeddings,valid_embeddings,similarity)
        return (optimizer,loss,distributed_embedding, dis_embed, structure_embedding, struct_embedd)

def main():

    graphfile = "../edgelist.graph"
    graph = tf.Graph()
    g = ti.read_graph_from_edgelist(graphfile)

    node_size   = g.number_of_nodes()
    struct_size = g.number_of_nodes()
    dim_size    = 200
    num_sampled = 100
    batch_size  = 1

    dline = DLINE()

    dline.load_graph_into_DLINE(g)

    with graph.as_default():
        valid_examples = np.zeros((5,5))
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None,1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)        


        #build_DLINE(node_size, struct_size, dim_size, num_sampled, train_inputs, train_labels, valid_dataset)
        (optimizer, loss, de, tde, se, tse) = dline.build_DLINE(node_size, struct_size, dim_size, num_sampled, 
            train_inputs, train_labels, valid_dataset)
        # Uncomment if you have any summary operations
        #merged = tf.merge_all_summaries()

    with tf.Session(graph=graph) as sess:
        #writer = tf.train.SummaryWriter("/tmp/worklogs", sess.graph_def)
        tf.initialize_all_variables().run()

    
        for t in dline.tuple_generator(train_inputs,train_labels):
            result = sess.run([optimizer, loss],t)

        return result
if __name__ == "__main__":
    main()
