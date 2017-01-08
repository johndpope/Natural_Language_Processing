"""End to End Memory Network (layer-wise)"""
from __future__ import division   
import numpy as np
import tensorflow as tf

def posEncoding(embedding_size,sentence_size):
    l=np.zeros((embedding_size,sentence_size),dtype=np.float32)
    for k in xrange(1,embedding_size+1):
        for j in xrange(1,sentence_size+1):
            l[k-1,j-1]=(1-j/sentence_size)-(k/embedding_size)*(1-2*j/sentence_size)
    return l
# l is embedding X sentence

def add_nil_column(embed_mat):
    #the first column should be all zero (represent nil), and should not be trained
    z=tf.convert_to_tensor(embed_mat)
    embed_size=tf.shape(z)[0]
    nil_column=tf.zeros(tf.pack([embed_size,1]))
    return tf.concat(1,[nil_column,tf.slice(embed_mat,[0,1],[-1,-1])])

class Memn2n(object):
    def __init__(self,batch_size,sentence_size,memory_size,embedding_size,vocabulary_size,
                session=tf.Session(),
                hops=3,
                max_grad_norm=40.0,
                nonLinear=None,
                initializer=tf.random_normal_initializer(stddev=0.1),
                optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
                encoding=posEncoding):
        self.batch_size=batch_size
        self.sentence_size=sentence_size
        self.memory_size=memory_size
        self.embedding_size=embedding_size
        self.vocabulary_size=vocabulary_size
        self.hops=hops
        self.max_grad_norm=max_grad_norm
        self.nonLinear=nonLinear
        self.init=initializer
        self.op=optimizer
        
        #some initial function
        self.build_input()
        self.build_parameter_matrix()
        self.posEncoding=tf.constant(posEncoding(self.embedding_size,self.sentence_size),name="pos_encoding")
        
        #inference, cost function
        res_vocabulary=self.inference(self.story,self.query)
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(res_vocabulary,tf.cast(self.answer,tf.float32))
        cross_entropy_sum=tf.reduce_sum(cross_entropy)
        loss=cross_entropy_sum
        
        #calculate and apply gradient (minimize=compute+apply)
        #A list of (gradient, variable) pairs
        gradient_variable=self.op.compute_gradients(loss)
        gradient_variable=[(tf.clip_by_norm(u,self.max_grad_norm),v) for u,v in gradient_variable]
        #can also add gradient noise to improve accuracy!
        gradient_variable_with_nil=[]
        for u,v in gradient_variable:
            if v.name in self.nil_name:
                gradient_variable_with_nil.append((add_nil_column(u),v))
            else:
                gradient_variable_with_nil.append((u,v))
        train_op=self.op.apply_gradients(gradient_variable_with_nil)
        
        #predict, return the number index of result
        predict_op=tf.arg_max(res_vocabulary,1)
        predict_proba_op=tf.nn.softmax(res_vocabulary)
        predict_log_proba_op=tf.log(predict_proba_op)
        
        self.loss_op=loss
        self.train_op=train_op
        self.predict_op=predict_op
        self.predict_proba_op=predict_proba_op
        self.predict_log_proba_op=predict_log_proba_op
        init_op=tf.initialize_all_variables()
        self.sess=session
        self.sess.run(init_op)
        
        
    def build_input(self):
        self.story=tf.placeholder(tf.int32,[None,self.sentence_size,self.memory_size],name="story")
        self.query=tf.placeholder(tf.int32,[None,self.sentence_size],name="query")
        self.answer=tf.placeholder(tf.int32,[None,self.vocabulary_size],name="answer")
    def build_parameter_matrix(self):
        with tf.variable_scope("parameter_matrix") as scope:
            nil_embedding=tf.zeros([self.embedding_size,1])
            #vocabulary size has included nil column, so here minus 1.
            A=tf.concat(1,[nil_embedding,self.init([self.embedding_size,self.vocabulary_size-1])])
            B=tf.concat(1,[nil_embedding,self.init([self.embedding_size,self.vocabulary_size-1])])
            #here set C the same with A
            self.A=tf.Variable(A,name='A')
            self.B=tf.Variable(B,name='B')
            self.H=tf.Variable(self.init([self.embedding_size,self.embedding_size]),name='H')
            self.W=tf.Variable(self.init([self.embedding_size,self.vocabulary_size]),name='W')
            #TA is related to temporal coding
            self.TA=tf.Variable(self.init([self.embedding_size,self.memory_size]),name='TA')
        self.nil_name=set([self.A.name,self.B.name])
    
    #the dimension should be validated using data!!!!!!!!!!!!!
    #story-> (one_batch_size,max_story_size,max_sentence_size)
    def inference(self,story,query): 
        q_embed=tf.nn.embedding_lookup(tf.transpose(self.B),query) #(batch_size,sentences_size,embedding_size)
        u_0=tf.reduce_sum(q_embed*tf.transpose(self.posEncoding),1) #(batch_size,embedding_size)   
        u=[u_0]
        for _ in range(self.hops):
            m_embed=tf.nn.embedding_lookup(tf.transpose(self.A),story) #batch_size X sentences_size X memory_size X embedding_size
            m=tf.transpose(tf.reduce_sum(tf.transpose(m_embed,[0,2,3,1])*self.posEncoding,3))+tf.expand_dims(self.TA,-1) 
            m=tf.transpose(m,[2,0,1])
            #m-> batch X embedding_size X memory_size  
            u_temp=tf.expand_dims(u[-1],-1) #(batch_size,embedding_size,1)
            prob=tf.nn.softmax(tf.reduce_sum(u_temp*m,1)) #(batch_size,memory_size)
            prob_temp=tf.expand_dims(prob,1) #(batch_size,1,memory_size)
            #here we set A and C are the same, so c=m
            c=m
            o=tf.reduce_sum(c*prob_temp,2) #(batch_size,embedding_size)
            u_next=tf.matmul(u[-1],self.H)+o #(batch_size,embedding_size)
            if self.nonLinear:
                u_next=nonLinear(u_next)
            u.append(u_next)
        return tf.matmul(u[-1],self.W) #(batch_size,vlocabulary_size)
    
    def train(self,story,query,answer):
        feed_dict={self.story:story,self.query:query,self.answer:answer}
        loss,_=self.sess.run([self.loss_op,self.train_op],feed_dict=feed_dict)
        return loss
    def predict(self,story,query):
        feed_dict={self.story:story,self.query:query}
        return self.sess.run(self.predict_op,feed_dict=feed_dict)
    def predict_proba(self,story,query):
        feed_dict={self.story:story,self.query:query}
        return self.sess.run(self.predict_proba_op,feed_dict=feed_dict)
    def predict_log_proba(self,story,query):
        feed_dict={self.story:story,self.query:query}
        return self.sess.run(self.predict_log_proba_op,feed_dict=feed_dict)