import tensorflow as tf
from tensorflow.contrib.learn.python.learn.preprocessing import text
import numpy as np
import cnn
import data_dealer

tf.flags.DEFINE_integer("epoch_size",10,"Default 10")
tf.flags.DEFINE_integer("batch_size",300,"Default 300")
tf.flags.DEFINE_integer("evaluate_step",50,"Evaluate each 50[default] global steps")
tf.flags.DEFINE_integer("embedding_size",200,"Default 200")
tf.flags.DEFINE_string("filter_size",'2,3,4',"Default '2,3,4' ")
tf.flags.DEFINE_integer("filter_num",3,"Filter numbers for each kind of filter, default 3")
tf.flags.DEFINE_float("keep_prob",0.7,"Probability of keep neuron when dropout, default 0.7")
tf.flags.DEFINE_integer("topic_num",12,"The number of different news topics, it depends on news corpus.")
tf.flags.DEFINE_bool("shuffle_input",True,"Default True")
tf.flags.DEFINE_float("train_dev_split_ratio",0.98,"Default 0.98, 98% is training data, 2% is development data")

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()

for para,val in FLAGS.__flags.items():
    print("parameter %s: %s"%(para,val))

print("Loading news and topic...")
all_urls, all_titles, all_news=data_dealer.import_data()
#data is a dictionary, comprised of health, auto, business, it, sports, learning, news, yule 10001 respectively.
data=data_dealer.subData(all_urls, all_titles, all_news)

health=zip(data['health'],np.ones([10001,1]))
auto=zip(data['auto'],2*np.ones([10001,1]))
business=zip(data['business'],3*np.ones([10001,1]))
x_news=data['health']+data['auto']+data['business']
y_label=[0]*10001+[1]*10001+[2]*10001

from tensorflow.contrib import learn
max_news_length=max([len(x.split(" ")) for x in x_news])
words_to_num=learn.preprocessing.VocabularyProcessor(max_news_length)
print('Maximal length in all news: %s' % max_news_length)
x_nums=np.array(list(words_to_num.fit_transform(x_news)))
vocabulary_size=len(words_to_num.vocabulary_)
print("There are %s Chinese vocabulary in all the news corpus." % vocabulary_size)
#processor.reverse(res)

if FLAGS.shuffle_input:
    print("Shuffle input data...")
    np.random.seed(1)
    new_indices=np.random.permutation(range(len(y_label)))
    x_nums=x_nums[new_indices]
    y_label=np.array(y_label)[new_indices]

print("Split input data into training and development part...")
x_train=x_nums[:FLAGS.train_dev_split_ratio*len(y_label),:]
y_train=y_label[:FLAGS.train_dev_split_ratio*len(y_label)]
x_dev=x_nums[FLAGS.train_dev_split_ratio*len(y_label):,:]
y_dev=y_label[FLAGS.train_dev_split_ratio*len(y_label):]


print("---------------Start training model...--------------------")
gra=tf.Graph()
with gra.as_default():
    sess=tf.Session()
    with sess.as_default():
        cnn=cnn.topicCNN(vocabulary_size=vocabulary_size,embedding_size=FLAGS.embedding_size,
                         filter_size=map(int,FLAGS.filter_size.split(',')),
                         filter_num=FLAGS.filter_num,max_news_size=max_news_length,topic_size=3)
        
        global_step=tf.Variable(0,name="global_step",trainable=False)
        optimizer=tf.train.AdamOptimizer()
        gradient_and_variable=optimizer.compute_gradients(cnn.loss)
        train_op=optimizer.apply_gradients(gradient_and_variable,global_step=global_step)
        
        sess.run(tf.initialize_all_variables())
        
        def train_one_step(x_batch,y_batch):
            feed_dict={cnn.input_news:x_batch,cnn.input_topic:y_batch,
                       cnn.dropout_keep_probability:FLAGS.keep_prob}
            _,step,loss,accuracy=sess.run([train_op,global_step,cnn.loss,cnn.accuracy],feed_dict)
            print("Train processing: step {}, loss {}, accuracy {}".format(step,loss,accuracy))
        
        def dev_one_step(x_batch,y_batch):
            feed_dict={cnn.input_news:x_batch,cnn.input_topic:y_batch,
                       cnn.dropout_keep_probability:1.0}
            step,loss,accuracy=sess.run([global_step,cnn.loss,cnn.accuracy],feed_dict)
            print("Dev processing: step {}, loss {}, accuracy {}".format(step,loss,accuracy))
        
        
        for epo in range(FLAGS.epoch_size):
            print('---------------Epoch: %s---------------' % epo)
            # input data in each epoch is not be permutated!
            for i in range(len(y_train)//FLAGS.batch_size):
                x_temp=x_train[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                y_temp=y_train[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                train_one_step(x_temp,y_temp)
                current_step=tf.train.global_step(sess,global_step)
                if current_step % FLAGS.evaluate_step==0:
                    print("Evalution start... at step %s"%current_step)
                    dev_one_step(x_dev,y_dev)
                    print("Evaluation end")


