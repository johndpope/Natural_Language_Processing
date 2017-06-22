# QA_System
Data: The (20) QA bAbI tasks, get from <https://research.fb.com/projects/babi/>
</br>
Example: </br>
1 Sandra travelled to the kitchen.</br>
2 Sandra travelled to the hallway.</br>
3 Where is Sandra? 	hallway	2</br>
</br>

Loss function: Cross-Entropy

<b>load_data.py</b>	-> Data preprocessing: Remove punctuation, extract story, question, answer part respectively.  Convert data into number-representation vector.
</br>
<b>memn2n.py</b>	-> End-to-End Memory Network Model.</br>
Position Encoding: Try to capture the word order information in a sentence.</br>
Temporal Encoding: Many of the QA tasks require some notion of temporal context, e.g. the model needs to understand that Sam is in the bedroom after he is in the kitchen.
</br>
<b>result.ipynb</b>	-> Train model, get classification accuracy about training data, validation data and test data.
</br>
![](mem.png)

Generally speaking, this project only deals with simple story text and one-word answer. Furthermore, as a very basic implementation, it couldn't deal with the out-of-vocabulary. But these problems are easy to solve, the goal of this project focus on understanding the principle of memory network. 

</br>
Reference: End-To-End Memory Networks, Sainbayar Sukhbaatar, 2015

Here is a much better implementation than mine: [https://github.com/carpedm20/MemN2N-tensorflow]
