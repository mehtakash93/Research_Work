import cPickle
import tensorflow as tf
import random
import numpy as np
import math

num_samples = 40

def sample_negatives(sample_Set, num_sampled):
    samples = []
    for _ in xrange(num_sampled):
        samples.append(random.sample(sample_Set, 1)[0])

    return (samples)

word2vecembed = cPickle.load(open('../data/word2vecEmbed.pkl', 'rW'))
word2vecDict=cPickle.load(open('../data/word2vecDict.pkl', 'rW'))
vocabulary_size=len(word2vecDict)
embedding_size=300

#This is the knowledge dict from sysnset(id) to actual indices
knowledgeDict=cPickle.load(open('../data/WN_synset2idx.pkl', 'rW'))
#This is the knowledge dict from actual indices to sysnset(id)
knowledgeRevDict=cPickle.load(open('../data/WN_idx2synset.pkl', 'rW'))
#This is the knowledge dict from synset id to acutal words
knowledgeDictConcept=cPickle.load(open('../data/WN_synset2concept.pkl', 'rW'))
#This is the entity embeddings from the knowledge graph
knowledgeEntity=cPickle.load(open('../data/KnowledgeEntity.pkl', 'rW'))
#This is the relations embeddings from the knowledge graph
knowledgeRel=cPickle.load(open('../data/KnowledgeRel.pkl', 'rW'))


# These are the actual triplets
inpl = cPickle.load(open('../data/WN-train-lhs.pkl', 'rW'))
inpr = cPickle.load(open('../data/WN-train-rhs.pkl', 'rW'))
inpo = cPickle.load(open('../data/WN-train-rel.pkl', 'rW'))
minInpo = min(inpo)
inpo[:] = [x-minInpo for x in inpo]
lset = set(inpl)
rset = set(inpr)
oset = set(inpo)

# Indexed list useful during negative sampling.
word_set=set(range(len(word2vecembed)))



#This is the word2vec embedding
Wembed=tf.Variable(word2vecembed)
#This is the knowledge entity embedding
Eembed=tf.Variable(knowledgeEntity)
#This is the knowledge relations embedding
Rembed=tf.Variable(knowledgeRel)
#For word2vec
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


#These are indices into the knowledge model embedding, and would always be present.
h = tf.placeholder(tf.int32, [1])
r = tf.placeholder(tf.int32, [1])
t = tf.placeholder(tf.int32, [1])




# This is a boolean, 0 if both head and tail from triple are not present in word2vec embedding, else 1
Iht=tf.placeholder(tf.float32, [1])
# This is a boolean, 0 if head is not present in word2vec embedding, else 1
Ih=tf.placeholder(tf.float32, [1])
# This is a boolean, 0 if tail is not present in word2vec embedding, else 1 
It=tf.placeholder(tf.float32, [1])




# These are the indices into word2vec embedding.
#If the word is not present i am selecting random entry
#but corresponding cost would be 0 as the "I" value would be zero.
hw=tf.placeholder(tf.int32, [1])
tw=tf.placeholder(tf.int32, [1])
hsamples = tf.placeholder(tf.int32, [num_samples])
rsamples = tf.placeholder(tf.int32, [num_samples])
tsamples = tf.placeholder(tf.int32, [num_samples])
w1samples = tf.placeholder(tf.int32, [num_samples])
w2samples = tf.placeholder(tf.int32, [num_samples])

tw2=tf.reshape(tw,[1,1])
#Lookup into knowledge embeddings
hrow = tf.nn.embedding_lookup(Eembed, h)
rrow = tf.nn.embedding_lookup(Rembed, r)
trow = tf.nn.embedding_lookup(Eembed, t)



#Lookup into word2vec embeddings
hwrow=tf.nn.embedding_lookup(Wembed, hw)
twrow=tf.nn.embedding_lookup(Wembed, tw)

hnegativerows = tf.nn.embedding_lookup(Eembed, hsamples)
rnegativerows = tf.nn.embedding_lookup(Rembed, rsamples)
tnegativerows = tf.nn.embedding_lookup(Eembed, tsamples)
w1negativerows = tf.nn.embedding_lookup(Wembed, w1samples)
w2negativerows = tf.nn.embedding_lookup(Wembed, w2samples)

 
#word2vec loss with negative sampling 
#only considering if bith head and tail present, hence multiplying by 0.
loss_w2w =tf.mul(tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, hwrow, tw2,num_samples, vocabulary_size)),Iht)


distance = lambda h,r,t: tf.sub(tf.constant(7, dtype="float32"),
    tf.mul(tf.constant(0.5, dtype="float32"),tf.reduce_sum(tf.pow(tf.sub(tf.add(h,r),t),2),1)))


#cost for knwoledge
positivez = distance(hrow, rrow, trow)
negativez = tf.neg(distance(hnegativerows, rnegativerows, tnegativerows))
cost_knowledge=cost = tf.neg(tf.log(tf.sigmoid(positivez)))-tf.reduce_sum(tf.log(tf.sigmoid(negativez)))

#Total cost
#cost using both head and tail from word2vec embeddings
#would be zero if either head or tail not present in word2vec embeddings as Iht would be 0

cost1=tf.mul(distance(hwrow, rrow, twrow),Iht)
cost1_neg = tf.neg(distance(w1negativerows, rnegativerows, w2negativerows))
#cost using head from word2vec embeddings
cost2=tf.mul(distance(hwrow, rrow, trow),Ih)
cost2_neg = tf.neg(distance(w1negativerows, rnegativerows, tnegativerows))
#cost using tail from word2vec embeddings
cost3=tf.mul(distance(hrow, rrow, twrow),It)  
cost3_neg = tf.neg(distance(hnegativerows, rnegativerows, w2negativerows)) 

#total cost without negative sampling for now.
cost_total=tf.sub(tf.add(tf.add(tf.neg(tf.log(tf.sigmoid(tf.add(cost1,tf.add(cost2,cost3))))),loss_w2w),cost_knowledge),tf.add(tf.reduce_sum(tf.log(tf.sigmoid(cost1_neg))),tf.reduce_sum(tf.log(tf.sigmoid(cost2_neg)))-tf.reduce_sum(tf.log(tf.sigmoid(cost3_neg)))))


train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost_total)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(100):
    idx = random.sample(range(len(inpl)), 1)
    #Taking random indices for negative sampling from knowledge graph
    hs = sample_negatives(lset, num_samples)
    os = sample_negatives(oset, num_samples)
    rs = sample_negatives(rset, num_samples)

    #negative samples from word2vec
    w1 = sample_negatives(word_set, num_samples)
    w2 = sample_negatives(word_set, num_samples)
    


    
    #Getting sysnet of the head and tail
    hwordsynset=knowledgeRevDict[inpl[idx[0]]]
    twordsynset=knowledgeRevDict[inpr[idx[0]]]

    #Getting the corresponding word
    hword=knowledgeDictConcept[hwordsynset]
    tword=knowledgeDictConcept[twordsynset]

    #This is boolean which go as input to It, Ih and Iht correspondingly.
    htemp=0
    ttemp=0
    httemp=0
  
    # These would be indices into the word2vec embeddings if word is present.
    hindex=1
    tindex=1

    if hword in word2vecDict:
    	htemp=1
    	hindex=word2vecDict[hword]
   	if tword in word2vecDict:
		ttemp=1
		tindex=word2vecDict[tword]
    if htemp==1 and ttemp==1:
    	httemp=1	


    [_,c]=sess.run([train_step,cost], 
        feed_dict={h: [inpl[idx[0]]], r: [inpo[idx[0]]], t: [inpr[idx[0]]],Iht:[httemp],Ih:[htemp],It:[ttemp],hw:[hindex],tw:[tindex], hsamples: hs, rsamples: os, tsamples: rs,w1samples:w1,w2samples:w2})    
    print "Step {}, Cost {}".format(i,c)


