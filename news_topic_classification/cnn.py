import tensorflow as tf
import numpy as np

class topicCNN(object):
    def __init__(self, vocabulary_size, embedding_size, filter_size, filter_num, max_news_size, topic_size):
        #input information
        self.input_news=tf.placeholder(tf.int64, [None, max_news_size], name='input_news')
        self.input_topic=tf.placeholder(tf.int64, [None], name='input_topic')
        #embedding size, notice that GPU cannot used in embedding.
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding_matrix=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1,1), name='embed_matrix')
            #self.embedding_news : [None(batch_size), max_news_size, embedding_size]
            self.embedding_news=tf.nn.embedding_lookup(embedding_matrix,self.input_news)
            #but the conv2d operation need 4-d tensor: [None(batch_size), max_news_size, embedding_size, channel]
            self.embedding_news=tf.expand_dims(self.embedding_news,-1)
        
        #covolutional layer and max_pool layer
        res_after_pool=[]
        for index,temp_filter_size in enumerate(filter_size):
            with tf.name_scope('convolution_pool_with_filter_size_%s' % index):
                filter_shape=[temp_filter_size,embedding_size,1,filter_num]
                w=tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='w')
                b=tf.Variable(tf.constant(0.1, shape=[filter_num]), name='b')
                #VALID denotes narrow convolution, without padding the edges.
                conv=tf.nn.conv2d(self.embedding_news,w,strides=[1,1,1,1],padding='VALID',name='convolution_layer')
                relu_conv=tf.nn.relu(tf.nn.bias_add(conv,b), name='relu')
                pool=tf.nn.max_pool(relu_conv,ksize=[1,max_news_size-temp_filter_size+1,1,1],
                                    strides=[1,1,1,1],padding='VALID',name='pool_layer')
                #tensor shape after pool: [batch_size,1,1,filter_num]
                res_after_pool.append(pool)
        
        #concatenate all pooled features and add dropout
        total_filter_num=len(filter_size)*filter_num
        self.pooled_feature=tf.concat(3,res_after_pool) 
        self.pooled_feature=tf.reshape(self.pooled_feature,[-1,total_filter_num])
        #[batch_size,total_filter_num]
        #probability of dropout, only applied when training.
        self.dropout_keep_probability=tf.placeholder(tf.float32, name='dropout_keep_probability')
        with tf.name_scope('drop_out'):
            self.dropout_feature=tf.nn.dropout(self.pooled_feature,self.dropout_keep_probability)
        
        #unnormalized scores and prediction
        with tf.name_scope('output'):
            w=tf.Variable(tf.truncated_normal([total_filter_num,topic_size],stddev=0.1),name='w')
            b=tf.Variable(tf.constant(0.1,shape=[topic_size]),name='b')
            self.scores=tf.nn.xw_plus_b(self.dropout_feature,w,b,name='scores')
            self.prediction=tf.argmax(self.scores,1,name='prediction')
        
        #loss
        with tf.name_scope('loss'):
            losses=tf.nn.sparse_softmax_cross_entropy_with_logits(self.scores, self.input_topic)
            self.loss=tf.reduce_mean(losses,name='loss')
            
        #accuracy
        with tf.name_scope('accuracy'):
            correct_pre=tf.equal(self.prediction,self.input_topic)
            self.accuracy=tf.reduce_mean(tf.cast(correct_pre,'float'),name='accuracy')
        
