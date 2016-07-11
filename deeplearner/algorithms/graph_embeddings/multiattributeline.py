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

class MALINE(object):
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
        super(MALINE, self).__init__()
        self.graphs = {}
        self.pairs = {}
        self.optimizers = {}
        self.loss = {}
        self.embeddings = {}
        self.frozen_embeddings = {}
        self.node_indices = {}
        self.edge_indices = {}

    def load_graph_into_LINE(self,graph,name):
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
        self.graphs[name] = graph
        pairs = []
        indices_nodes = tu.build_vocab(graph.nodes())
        for e in graph.edges_iter():
            pairs.append((indices_nodes[e[0]], indices_nodes[e[1]]))
        self.pairs[name] = pairs
        self.node_indices[name] = indices_nodes.values()
        return pairs

    def load_graph_into_MALINE(self,graph,name, graph1name, graph2name):
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
        self.graphs[name] = graph
        pairs = []
        indices_edges = tu.build_vocab(graph.edges())
        for e in graph.edges_iter():
            pairs.append(((self.node_indices[graph1name][e[0]],self.node_indices[graph2name][e[1]])
                ,indices_edges[e]))

        self.pairs[name] = pairs
        self.edges_indices[name] = indices_edges.values()
        return pairs    

    def cast_to_input(self, pair, training_placeholder, test_placeholder):
        return {training_placeholder: [pair[0]], test_placeholder: [pair[1]]}

    def line_tuple_generator(self, name, training_placeholder, test_placeholder):
        """
            Generator object that allows the class to feed the input/output pairs
            into the session.run instance inside the tensorflow pipeline
        """
        for i in self.pairs[name]:
            yield self.cast_to_input(i, training_placeholder, test_placeholder)

    def shuffled_line_tuple_generator(self, name, training_placeholder, test_placeholder):
        """
            Generator object that allows the class to feed randomly shuffled input/output pairs
            into the session.run instance inside the tensorflow pipeline
        """
        npr.shuffle(self.pairs[name])
        return self.line_tuple_generator(name, training_placeholder,test_placeholder)

    def create_line_negative_samples(self, node_indices, num_samples, negativex_placeholder, negativey_placeholder):
        """
            Generator object that creates the input set for the negative sampling portion of the
            algorithm
        """
        negativex = npr.choice(node_indices, num_samples)
        negativey = npr.choice(node_indices, num_samples)
        return {negativex_placeholder: negativex, negativey_placeholder: negativey}

    def create_shuffled_line_joint_generator(self, name, training_placeholder, test_placeholder,
        node_indices, num_samples, negativex_placeholder, negativey_placeholder):
        """
            Generator object that creates a feed dictionary that contains the training
            input, training label, and negative samples
        """
        for t in self.shuffled_line_tuple_generator(name, training_placeholder, test_placeholder):
            t.update(
                self.create_line_negative_samples(node_indices, num_samples, 
                    negativex_placeholder, negativey_placeholder))
            yield t

    def create_shuffled_maline_joint_generator(self, name, graph1name, graph2name, left_placeholder,
        edge_placeholder, right_placeholder, num_samples, lnegative_placeholder, enegative_placeholder,
        rnegative_placeholder):
        """
            Generator object that creates a feed dictionary that contains the training
            input, training label, and negative samples
        """
        for ((l,r),e) in self.pairs[name]:
            lnegative = npr.choice(self.node_indices[graph1name], num_samples)
            enegative = npr.choice(self.edge_indices[name], num_samples)
            rnegative = npr.choice(self.node_indices[graph2name], num_samples)

            t = {left_placeholder: l, edge_placeholder: e, right_placeholder: r,
                lnegative_placeholder: lnegative, enegative_placeholder: enegative, 
                rnegative_placeholder:rnegative}

            yield t

    def build_LINE(self, name, node_size, dim_size, num_sampled):
        """
            This method will build a tensorflow computation graph that will embed the graph
            loaded under the given name
        """
        with tf.name_scope(name):
            valid_examples = np.zeros((5,5))
            train_inputs = tf.placeholder(tf.int32, shape=[None])
            train_labels = tf.placeholder(tf.int32, shape=[None])
            negativex = tf.placeholder(tf.int32, shape=[num_sampled])
            negativey = tf.placeholder(tf.int32, shape=[num_sampled])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)        


            #self,node_size, dim_size, num_sampled, train_inputs, train_labels, valid_dataset
            (optimizer, loss, embeddings) = self.build_LINE_NN(node_size, dim_size,
                train_inputs, train_labels, negativex, negativey, valid_dataset)

            #self.optimizer[name] = optimizer
            #self.loss[name] = loss
            #self.embeddings[name] = embeddings

        return (train_inputs, train_labels, negativex, negativey, optimizer, loss, embeddings)
    def build_LINE_NN(self,node_size, dim_size, train_inputs, train_labels, negativex, negativey, valid_dataset):
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
        bias = tu.create_bias(dim_size, 0, "context")

        # Look up embeddings for inputs, output and negative samples.
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        output = tf.nn.embedding_lookup(embeddings, train_labels)

        negative_input = tf.nn.embedding_lookup(embeddings, negativex)
        negative_output = tf.nn.embedding_lookup(embeddings, negativey)

        negativelog = lambda x,y,b: tf.neg(tf.log(tf.sigmoid(tf.add(tf.mul(x,y), b))))

        # Compute the approximated negative likelihood for a model
        loss = tf.reduce_sum(negativelog(embed, output, bias)) + tf.reduce_sum(negativelog(negative_input, negative_output, bias))

        optimizer = tf.train.GradientDescentOptimizer(.1).minimize(loss)

        return (optimizer,loss,embeddings)

    def add_frozen_embedding(self, name, value):
        self.frozen_embeddings[name] = value

    def build_MALINE(self, left_embedding, right_embedding, left_dim, right_dim, 
        edge_size, dim_size, num_sampled):
        """
            This method will build a tensorflow computation graph that will embed the graph
            loaded under the given name
        """
        with tf.name_scope("maline"):
            valid_examples = np.zeros((5,5))
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)        

            left_embedding = tf.constant(left_embedding)
            right_embedding = tf.constant(right_embedding)

            left_index = tf.placeholder(tf.int32, shape=[1])
            right_index = tf.placeholder(tf.int32, shape=[1])
            edge_index = tf.placeholder(tf.int32, shape=[1])

            lnegative = tf.placeholder(tf.int32, shape=[num_sampled])
            enegative = tf.placeholder(tf.int32, shape=[num_sampled])
            rnegative = tf.placeholder(tf.int32, shape=[num_sampled])


            #self,node_size, dim_size, num_sampled, train_inputs, train_labels, valid_dataset
            (optimizer, loss, lembeddings, eembeddings,rembedding) = self.build_MALINE_NN(edge_size, dim_size,
                left_embedding, right_embedding, left_dim, right_dim, 
                left_index, right_index, edge_index, lnegative, enegative, 
                rnegative, valid_dataset)

        return (optimizer, loss, lembeddings, eembeddings,rembedding, 
            left_index, edge_index, right_index, lnegative, enegative, rnegative)

    def build_MALINE_NN(self, edge_size, dim_size, left_embedding, right_embedding, left_dim, 
        right_dim, left_index, right_index, edge_index, lnegative, enegative, 
        rnegative, valid_dataset):
        """
            This method will build a tensorflow computation graph representing a triplet embedding to learn 
            joint embeddings. This function uses the default computation graph. It does not create a new
            computation graph, and therefore the default computation graph needs to be created before this
            function is ran inside it's with statement

            For more information, please read this:
                https://www.tensorflow.org/versions/v0.6.0/tutorials/mnist/tf/index.html

            This code is a near exact copy from https://github.com/tensorflow/tensorflow/blob/0.6.0/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
            Specifically, lines 152-170

            In addition, for information on the formulation of the distance function, please see
            the paper (Wang, Zhen, et al. "Knowledge Graph and Text Jointly Embedding." EMNLP. 2014.)

            Parameters:
                edge_size: The number of edges connecting graphs 1 and 2
                dim_size: The number of dimensions of the edge and projection embedding
                left_embedding: The tensorflow object representing the left embedding
                right_embedding: The tensorflow object representing the left embedding
                left_dim: A tuple representing the dimensions of the left embedding
                right_dim: A tuple representing the dimensions of the left embedding
                left_index:The placeholder representing the left input
                right_index: The placeholder representing the right input
                edge_index:The placeholder representing the edge input 
                lnegative:The placeholder representing the left negative input 
                enegative: The placeholder representing the left negative input 
                rnegative:The placeholder representing the left negative input
                validation_dataset: The ohe vectors for node's whose neighor to node similarity
                    you want to check during the validation set
            Return:
                The tensorflow node objects for:
                    the optimizer
                    loss function
                    Projection matrix for the left node
                    Embedding matrix for the edge relations

        """

        edge_embedding = tu.create_rweight(edge_size, dim_size, "edge_embedding",.1)
        left_projection = tu.create_rweight(left_dim[1], dim_size)
        right_projection = tu.create_rweight(right_dim[1], dim_size)

        # Look up embeddings for inputs tuples
        leftembed = tf.nn.embedding_lookup(left_embedding, left_index)
        edgeembed = tf.nn.embedding_lookup(edge_embedding, edge_index)
        rightembed = tf.nn.embedding_lookup(right_embedding, right_index)

        # Look up the embeddings for the negative samples
        leftnegativeembed = tf.nn.embedding_lookup(left_embedding, lnegative)
        rightnegativeembed = tf.nn.embedding_lookup(right_embedding, rnegative)
        edgenegativeembed = tf.nn.embedding_lookup(edge_embedding, enegative)

        # Project the left and right embeddings into the new space
        leftproject = tf.matmul(leftembed, left_projection)
        rightproject = tf.matmul(rightembed, right_projection)

        leftnegativeproject = tf.matmul(leftnegativeembed, left_projection)
        rightnegativeproject = tf.matmul(rightnegativeembed, right_projection)

        distance = lambda h,r,t: tf.sub(tf.constant(7, dtype="float32"),
            tf.mul(tf.constant(0.5, dtype="float32"),tf.reduce_sum(tf.pow(tf.sub(tf.add(h,r),t),2),1)))

        negativelog = lambda z: tf.neg(tf.log(tf.sigmoid(z)))
        positivez = distance(leftproject, edgeembed, rightproject)
        negativez = distance(leftnegativeproject, edgenegativeembed, rightnegativeproject)

        loss = negativelog(positivez) + tf.reduce_sum(negativez)

        optimizer = tf.train.GradientDescentOptimizer(.1).minimize(loss)

        return (optimizer,loss,left_projection, edgeembed, right_projection)

    def train_LINE(self, graphfile, name, dim_size, num_sampled, num_iterations):
        """
        """
        graph = tf.Graph()
        g = ti.read_nxgraph_from_edgelist(graphfile)

        node_size = g.number_of_nodes()

        self.load_graph_into_LINE(g, name)
        with graph.as_default():
            tf.set_random_seed(1)

            (train_inputs, train_labels, negativex, 
                negativey, optimizer, loss, embeddings) = self.build_LINE(name, node_size,
                 dim_size, num_sampled)

        with tf.Session(graph=graph) as sess:
            #writer = tf.train.SummaryWriter("/tmp/worklogs", sess.graph_def)
            tf.initialize_all_variables().run()

            print "Starting Graphs"
            print "-------------------------------"
            print "Number of nodes:", g.number_of_nodes()
            print "Number of edges: ", g.number_of_edges()
            print "-------------------------------"
            output_steps = 10000
            current_step = 1
            loss_total = 0

            current_embedding = np.zeros(1)
            for i in xrange(num_iterations):
                print "Starting iteration {}:".format(i)

                generator = self.create_shuffled_line_joint_generator(name, train_inputs, train_labels,
                    g.nodes(), num_sampled, negativex, negativey)

                for t in generator:
                    _,computed_loss, current_embedding= sess.run([optimizer, loss, embeddings],t)

                    loss_total += computed_loss
                    if current_step % output_steps == 0:
                        print "Current Tuple: {}, Loss {}".format(current_step, loss_total/output_steps)
                        loss_total = 0
                    current_step += 1

                loss_total = 0
                current_step = 0

            self.add_frozen_embedding(name, current_embedding)

    def train_MALINE(self, graphfile, name, graph1name, graph2name, dim_size, num_sampled, num_iterations):
        """
        """
        graph = tf.Graph()
        g = ti.read_nxgraph_from_edgelist(graphfile)

        edge_size = g.number_of_edges()
        left_embedding = self.frozen_embeddings[graph1name]
        right_embedding = self.frozen_embeddings[graph2name]

        left_dim = left_embedding.shape
        right_dim = right_embedding.shape

        self.load_graph_into_MALINE(g, name)
        with graph.as_default():
            tf.set_random_seed(1)

            (optimizer, loss, lembeddings, eembeddings, rembeddings, 
            left_index, edge_index, right_index, 
            lnegative, enegative, rnegative) = self.build_MALINE(left_embedding, 
            right_embedding, left_dim, right_dim, edge_size, dim_size, num_sampled)

        with tf.Session(graph=graph) as sess:
            #writer = tf.train.SummaryWriter("/tmp/worklogs", sess.graph_def)
            tf.initialize_all_variables().run()

            print "Starting Graphs"
            print "-------------------------------"
            print "Number of nodes:", g.number_of_nodes()
            print "Number of edges: ", g.number_of_edges()
            print "-------------------------------"
            output_steps = 10000
            current_step = 1
            loss_total = 0

            current_embedding = np.zeros(1)
            for i in xrange(num_iterations):
                print "Starting iteration {}:".format(i)

                generator = self.create_shuffled_maline_joint_generator(name, graph1name, graph2name, left_index,
                edge_index, right_index, num_sampled, lnegative, enegative, rnegative)

                for t in generator:
                    _,computed_loss, left_project, edge_project, right_project= sess.run([optimizer, loss, lembeddings, eembeddings, rembeddings],t)

                    loss_total += computed_loss
                    if current_step % output_steps == 0:
                        print "Current Tuple: {}, Loss {}".format(current_step, loss_total/output_steps)
                        loss_total = 0

                    current_step += 1

                loss_total = 0
                current_step = 0

            self.add_frozen_embedding(name, current_embedding)

def main():

    graph1file = "../../../data/facebook/facebook_combined.txt"
    graph2file = ""
    #bipartitefile = ""

    #graph = tf.Graph()

    #g1 = ti.read_nxgraph_from_edgelist(graph1file)
    #g2 = ti.read_nxgraph_from_edgelist(graph2file)
    #bg = ti.read_nxgraph_from_edgelist(bipartitefile)

    dim1_size     = 256
    dim2_size     = 256
    #dim_edge_size = 100
    num_sampled = 10
    num_iterations = 1

    maline = MALINE()

    maline.train_LINE(graph1file, "source1", dim1_size, num_sampled, num_iterations)
    #maline.train_LINE(graph2file, "source2", dim2_size, num_sampled, num_iterations)

    #maline.load_graph_into_MALINE(bg, "mapping")
    #tsne.create_TSNE_plotfile(current_embedding, "line.png")
    #np.save("line_embedding.npy", current_embedding)

if __name__ == "__main__":
    main()