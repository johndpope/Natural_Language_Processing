# news_topic_classification

Data source: Sogou Labs news corpus

Data Format: news_url, page_ID, page_title, page_content (all included in HTML label)

Classification Label: auto, health, sport, technology, etc.

Firstly clean the raw data to standard format so as to accepted by Tensorflow. Limited by computer capacity, select three sub-topics: auto, health and business, each of topics includes 10,000 data.

Set training parameters epoch_size, batch_size, embedding_size, drop_out_probability, train_test_split_ratio and CNN parameters filter_size, filter_num.

![](pic/cnn.png)

CNN is so fast that take only 3 minutes to finish one epoch and get an accuracy of 88%.

![](pic/cnn_accuracy.png)


Set RNN parameters hidden_size(memory_size).

![](pic/lstm.png)

RNN is relative slower, it takes 10 minutes to finish one epoch, but attains 92% accuracy.

CNN performs very good in computer vision because it ignore object positions and extract local feature. In natural language processing, RNN simulates human’s from-left-to-right reading habit but CNN smooth and ignore word order, that’s why RNN has higher accuracy.