# ==============================================================================
# Name: line.py
# Author: Dakota Medd
# Version: 1.0
# Description: This file contains the LINE class, which manages code pertaining
# to the LINE algorithm.
# ==============================================================================

import sys
sys.path.append("../../../")
sys.path.append("../../")
import numpy as np
import numpy.random as npr

import tensorflow as tf
import func.tensorflow_io as ti
import func.tensorflow_utils as tu
import visualization.tsne_plot as tsne

class LINE(object):
    """
        LINE is a graph embedding algorithm that attempts to find
        a lower dimensional representation of a graph by attempting
        to learn an auto-encoder that learns the distribution of
        nodes in the graph.

        This object works in conjunction with tensorflow (https://www.tensorflow.org/)
        to load and compute these embeddings. Tensorflow needs to be installed to run
        this algorithm.
    """

    def __init__(self):
        super(LINE, self).__init__()
        self.pairs = []

    def load_graph_into_LINE(self,graph):
        """
            Takes in a networkx Graph object and converts the edges into
            tuples of one-hot-encoded vectors, representing the input
            and expected output used to train LINE

            Parameters:
                graph: A networkx graph object
            Return:
                A list of tuples, denoting edges with the node id being
                their one-hot encoded representation
        """
        pairs = []
        indices_nodes = tu.build_vocab(graph.nodes())
        for e in graph.edges_iter():
            pairs.append((indices_nodes[e[0]], [indices_nodes[e[1]]]))

        self.pairs = pairs
        return pairs

    def cast_to_input(self, pair, training_placeholder, test_placeholder):
        return {training_placeholder: [pair[0]], test_placeholder: [pair[1]]}

    def tuple_generator(self,training_placeholder, test_placeholder):
        """
            Generator object that allows the class to feed the input/output pairs
            into the session.run instance inside the tensorflow pipeline
        """
        for i in self.pairs:
            yield self.cast_to_input(i, training_placeholder, test_placeholder)

    def shuffled_tuple_generator(self,training_placeholder, test_placeholder):
        """
            Generator object that allows the class to feed randomly shuffled input/output pairs
            into the session.run instance inside the tensorflow pipeline
        """
        npr.shuffle(self.pairs)
        return self.tuple_generator(training_placeholder,test_placeholder)


    def build_LINE(self,node_size, dim_size, num_sampled, train_inputs, train_labels, valid_dataset):
        """
            This method will build a tensorflow computation graph representing the auto-encoder used by LINE 
            to learn embeddings. This function uses the default computation graph. It does not create a new
            computation graph, and therefore the default computation graph needs to be created before this
            function is ran inside it's with statement

            For more information, please read this:
                https://www.tensorflow.org/versions/v0.6.0/tutorials/mnist/tf/index.html

            This code is a near exact copy from https://github.com/tensorflow/tensorflow/blob/0.6.0/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
            Specifically, lines 152-170

            Parameters:
                node_size: The number of nodes inside the graph
                dim_size: The number of dimensions for the learned embedding
                num_sampled: The number of edges to sample for negative-sampling (read paper)
                train_inputs: The placeholder tensorflow node that represents the input node
                train_labels: The placeholder tensorflow node that represents the output node
                validation_dataset: The ohe vectors for node's whose neighor to node similarity
                    you want to check during the validation set
            Return:
                The tensorflow node objects for:
                    the optimizer
                    loss function
                    normalized_embeddings
                    valid_embeddings
                    similarity computation

        """

        embeddings = tu.create_rweight(node_size, dim_size, "embedding",.1)
        bias = tu.create_bias(node_size, 0, "context")

        # Look up embeddings for inputs.
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        output_weights = embeddings

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        nce_loss = tf.nn.nce_loss(output_weights, bias, 
            embed, train_labels, num_sampled, node_size)

        loss = tf.reduce_mean(nce_loss)

        optimizer = tf.train.GradientDescentOptimizer(.1).minimize(loss)

        return (optimizer,loss,embeddings)

def main():

    graphfile = "../../../data/facebook/facebook_combined.txt"
    graph = tf.Graph()
    g = ti.read_nxgraph_from_edgelist(graphfile)

    node_size   = g.number_of_nodes()
    dim_size    = 256
    num_sampled = 1000
    num_iterations = 5

    line = LINE()

    line.load_graph_into_LINE(g)
    print "Starting"
    print "-------------------------------"
    print "Number of nodes:", node_size
    print "Number of edges: ", g.number_of_edges()
    print "-------------------------------"

    with graph.as_default():
        tf.set_random_seed(1)
        valid_examples = np.zeros((5,5))
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None,1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)        

        #self,node_size, dim_size, num_sampled, train_inputs, train_labels, valid_dataset
        (optimizer, loss, embeddings) = line.build_LINE(node_size, dim_size, num_sampled, 
            train_inputs, train_labels, valid_dataset)

        # Uncomment if you have any summary operations
        #merged = tf.merge_all_summaries()

    with tf.Session(graph=graph) as sess:
        #writer = tf.train.SummaryWriter("/tmp/worklogs", sess.graph_def)
        tf.initialize_all_variables().run()

        output_steps = 10000
        current_step = 1
        loss_total = 0

        current_embedding = np.zeros(1)
        for i in xrange(num_iterations):
            print "Starting iteration {}:".format(i)

            for t in line.shuffled_tuple_generator(train_inputs,train_labels):
                _,computed_loss, current_embedding= sess.run([optimizer, loss, embeddings],t)

                loss_total += computed_loss
                if current_step % output_steps == 0:
                    print "Current Tuple: {}, Loss {}".format(current_step, loss_total/output_steps)
                    loss_total = 0

                current_step += 1

            loss_total = 0
            current_step = 0

        tsne.create_TSNE_plotfile(current_embedding, "line.png")
        np.save("line_embedding.npy", current_embedding)

if __name__ == "__main__":
    main()