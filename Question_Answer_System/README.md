# QA_System
Data: The (20) QA bAbI tasks, get from <https://research.fb.com/projects/babi/>
</br>
Example: </br>
1 Sandra travelled to the kitchen.</br>
2 Sandra travelled to the hallway.</br>
3 Where is Sandra? 	hallway	2</br>
</br>
load_data.py	-> Data preprocessing: Remove punctuation, extract story, question, answer part respectively.  Convert data into number-representation vector.
</br>
memn2n.py	-> End-to-End Memory Network Model.</br>
Position Encoding: Try to capture the word order information in a sentence.</br>
Temporal Encoding: Many of the QA tasks require some notion of temporal context, e.g. the model needs to understand that Sam is in the bedroom after he is in the kitchen.
</br>
result.ipynb	-> Train model, get classification accuracy about training data, validation data and test data.
</br>
![](mem.png)
</br>
Reference: End-To-End Memory Networks, Sainbayar Sukhbaatar, 2015
