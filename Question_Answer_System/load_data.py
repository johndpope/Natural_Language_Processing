import os
import re
import numpy as np
#data_dir="/Users/shichangtai/Desktop/Natural Language Processing/QA system/tasks_1-20_v1-2/en"
#task_id=1
def load_data_from_file(data_dir,task_id,only_support_sentence=False):
    assert task_id<21 and task_id>0
    files=os.listdir(data_dir)
    files=[os.path.join(data_dir,f) for f in files]
    symbol='qa{}_'.format(task_id)
    train_files=[f for f in files if 'train' in f and symbol in f][0]
    test_files=[f for f in files if 'test' in f and symbol in f][0]
    train_data=convert_text_to_data(train_files,only_support_sentence)
    test_data=convert_text_to_data(test_files,only_support_sentence)
    return train_data,test_data

def tokenize(sentence):
    return [s.strip() for s in re.split('(\W+)?',sentence) if s.strip()]
def convert_text_to_data(data_file,only_support_sentence):
    with open(data_file) as f:
        lines=f.readlines()
        data=[]
        story=[]
        for line in lines:
            line=line.lower()
            index, content=line.split(' ',1)
            index=int(index)
            if index==1:
                story=[]
            if '\t' in content: # question
                ques,ans,support=content.split('\t')
                ques=tokenize(ques)
                ans=[ans]
                support_story=None
                if ques[-1]=='?':
                    ques=ques[:-1]
                if only_support_sentence:
                    support_story=map(int,support.split())
                    support_story=[story[i-1] for i in support_story]
                else:
                    #select all sentences
                    support_story=[i for i in story if i]
                data.append((support_story,ques,ans))
                story.append('') #question row is also a row in story, since it has index number.
            else:
                line_list=tokenize(content)
                if line_list[-1]=='.':
                    line_list=line_list[:-1]
                story.append(line_list)
        return data  # data -> [([story],[question],[answer])]
    
def convert_data_to_number_list(data,word_index,sentence_size,memory_size):
    #word_index does not include nil, we mark nil to 0
    S=[]
    Q=[]
    A=[]
    for story,question,answer in data:
        story_num=[]        
        for index, text in enumerate(story,1):
            zero_num=max(0,sentence_size-len(text))
            story_num.append([word_index[i] for i in text]+zero_num*[0])
        #if story number is more than memory size, preserve nearest stories.
        story_num=story_num[::-1][:memory_size][::-1]
        #if story number is less, pad zero
        zero_num=max(0,memory_size-len(story_num))
        for _ in range(zero_num):
            story_num.append([0]*sentence_size)
        #pad zero to question
        zero_num=max(0,sentence_size-len(question))
        question_num=[word_index[x] for x in question]+[0]*zero_num
        #one-hot output vocabulary
        res=np.zeros(len(word_index)+1)
        for item in answer:
            res[word_index[item]]=1
        S.append(story_num)
        Q.append(question_num)
        A.append(res)
    return np.array(S),np.array(Q),np.array(A)
# return a tuple (Story, Question, Answer)