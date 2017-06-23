from bs4 import BeautifulSoup
import numpy as np
import re
import jieba
#pure_data=BeautifulSoup(data,'lxml')
def import_data(data_path='/u/80/shic1/unix/Downloads/news_sohusite_xml.dat'):
    with open(data_path) as f:
        data=f.readlines()
    all_urls=data[1::6]
    all_titles=data[3::6]
    all_news=data[4::6]
    print('Data has beend imported!')
    return all_urls, all_titles, all_news

def convert_chinese(words):
    #remove html label
    pure_data = re.search('>(.*?)</',words).group(1).decode('GB18030')
    #remove all the emotion, punctuations or numbers except chinese words
    pure_text = re.sub(u'[^\u4e00-\u9fa5]','',pure_data)
    #token text to a list of vocabularies
    return ' '.join(jieba.cut(pure_text))

def extract_url_topic(url):
    return re.search('(\w*?)\.sohu',url).group(1)

#pick health, auto, business, it, sports, learning, news, yule 10000 respectively.
def subData(a,b,c):
    sub_data=dict()
    topics=['health','auto','business','it','sports','learning','news','yule']
    for i in topics:
        sub_data[i]=[]
    count=len(a)//10        
    for i in xrange(len(a)):
        if (i%count==0):
            print('Sub-data has imported %s percentage' % (float(i)/len(a)*100.0))
        if (extract_url_topic(a[i]) in topics):
            if(len(sub_data[extract_url_topic(a[i])])>10000):
                continue
            sub_data[extract_url_topic(a[i])].append(convert_chinese(c[i]))
    return sub_data
