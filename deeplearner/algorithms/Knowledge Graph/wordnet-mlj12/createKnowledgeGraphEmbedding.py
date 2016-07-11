import sys
import tensorflow as tf
from sklearn.datasets import load_boston
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize
import random as random

import cPickle

num_samples = 40
def sample_negatives(lset, rset, oset, num_sampled):
    right_samples = []
    left_samples = []
    relation_samples = []

    for _ in xrange(num_sampled):
        right_samples.append(random.sample(rset, 1)[0])
        left_samples.append(random.sample(lset, 1)[0])
        relation_samples.append(random.sample(oset, 1)[0])

    return (left_samples, relation_samples, right_samples)

inpl = cPickle.load(open('../data/WN-train-lhs.pkl', 'rW'))
inpr = cPickle.load(open('../data/WN-train-rhs.pkl', 'rW'))
inpo = cPickle.load(open('../data/WN-train-rel.pkl', 'rW'))

minInpo = min(inpo)
inpo[:] = [x-minInpo for x in inpo]

entityCount = len(set(inpl).union(set(inpr)))
relCount = len(set(inpo))

lset = set(inpl)
rset = set(inpr)
oset = set(inpo)

W = tf.Variable(tf.random_uniform([entityCount, 300],-.05,.05))
R = tf.Variable(tf.random_uniform([relCount, 300],-.05,.05))
#Wd = tf.Variable(tf.random_uniform([entityCount, 300],-.05,.05))

h = tf.placeholder(tf.int32, [1])
r = tf.placeholder(tf.int32, [1])
t = tf.placeholder(tf.int32, [1])

hsamples = tf.placeholder(tf.int32, [num_samples])
rsamples = tf.placeholder(tf.int32, [num_samples])
tsamples = tf.placeholder(tf.int32, [num_samples])


hrow = tf.nn.embedding_lookup(W, h)
rrow = tf.nn.embedding_lookup(R, r)
trow = tf.nn.embedding_lookup(W, t)

lnegativerows = tf.nn.embedding_lookup(W, hsamples)
rnegativerows = tf.nn.embedding_lookup(R, rsamples)
tnegativerows = tf.nn.embedding_lookup(W, tsamples)

distance = lambda h,r,t: tf.sub(tf.constant(7, dtype="float32"),
    tf.mul(tf.constant(0.5, dtype="float32"),tf.reduce_sum(tf.pow(tf.sub(tf.add(h,r),t),2),1)))

positivez = distance(hrow, rrow, trow)
negativez = tf.neg(distance(lnegativerows, rnegativerows, tnegativerows))
#z = tf.sub(tf.constant(7, dtype="float32"),tf.mul(tf.constant(0.5, dtype="float32"),tf.reduce_sum(tf.square(tf.sub(tf.add(hrow,rrow),trow)))))
#z = tf.sub(tf.constant(7, dtype="float32"),tf.mul(tf.constant(0.5, dtype="float32"),tf.reduce_sum(tf.sub(tf.add(hrow,rrow),trow))))
cost = tf.neg(tf.log(tf.sigmoid(positivez)))-tf.reduce_sum(tf.log(tf.sigmoid(negativez)))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(100):
    idx = random.sample(range(len(inpl)), 1)
    [hs, rs, ts] = sample_negatives(lset, rset, oset, num_samples)
    #print inpl[idx[0]]
    #print inpo[idx[0]]
    #print inpr[idx[0]]
    #idx = [10000]
    print 'Running with h:',inpl[idx[0]]
    [_, c, Eembed, Rembed] = sess.run([train_step, cost, W, R], 
        feed_dict={h: [inpl[idx[0]]], r: [inpo[idx[0]]], t: [inpr[idx[0]]], hsamples: hs, rsamples: rs, tsamples: ts})

    if i % 10:
        print "Step {}, Cost {}".format(i,c)

e = open('../data/KnowledgeEntity.pkl', 'w')
cPickle.dump(Eembed, e, -1)
e.close()
r = open('../data/KnowledgeRel.pkl', 'w')
cPickle.dump(Rembed, r, -1)
r.close()