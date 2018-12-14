# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 05:15:25 2018

@author: wenqi
"""

import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from keras.models import Model
import keras.models
from keras import backend as K

def extract_title(title):
    n = len(title)
    
    ex_title1 = None
    ex_title2 = None
    ex_title3 = None
    ex_title4 = None
    if title == None :
        return ex_title1,ex_title2,ex_title3,ex_title4
    ye=99999
    i = n-3
    while i in range(n):
        if (title[i]=='2')or(title[i]=='1')and(i+4<=n) :
            st = title[i:i+4]
            if st.isdigit():
                if (int(st)>1948)and(int(st)<2018):
                    ye = i
                    i = 99999
        i -= 1
    if ye != 99999:
        ex_title1 = title[:ye]
        ex_title2 = title[ye:ye+4]
        title = title[ye+4:]
    n = len(title)
    i = n-1
    brac=99999
    if title[i]==')':  
        while i in range(n):
            if title[i]=='(':
                brac = i
                i = 99999
            i -= 1
    else:
        ex_title3=title[0:]
    if brac != 99999:
        ex_title3 = title[0:brac]
        ex_title4=title[brac+1:n-1]
    
    ex_title1 = ex_title1.strip() if ex_title1 != None else None
    ex_title2 = ex_title2.strip() if ex_title2 != None else None
    ex_title3 = ex_title3.strip() if ex_title3 != None else None
    ex_title4 = ex_title4.strip() if ex_title4 != None else None
    return ex_title1,ex_title2,ex_title3,ex_title4
def find(sample,key,value,col_list):
    if value == None:
        sample['key'+'_nan']=1
        return sample
    flag =False
    n=len(key)
    for k in col_list:
        if (key == k[:n]) and (value==k[n+1:]):
            sample[k] = 1
            flag = True
            return sample
    other = key+'_other'
    if (not flag) and (other in col_list):
        sample[other] = 1
    return sample
        
def bow(sample,x,col_list):
    x = x.lower() 
    x = word_tokenize(x) 
    stop_words = set(stopwords.words('english'))
    ps=PorterStemmer()
    new_x=[ps.stem(stw) for stw in  x]
    x = [xx for xx in new_x if xx not in stop_words]
    
    m = len(x)
    
    for i in col_list:
        if i[len(i)-4:] == '_nlp':
            for j in range(m):
                if (x[j] == i[:len(i)-4]):
                    sample[i] += 1
    
    return sample
    
    
#Use None to file the blank of the list
def predict(info):
    title,points,describe,designation,variety,winery,taster_name,country,province=info[:]
            
    with open('data_format','rb') as f:
        sample = pickle.load(f) 
    col_list = sample.index 
    if points!=None :
        sample['points']=points
    else :
        sample['points']=88
    ex_title1,year,ex_title2,ex_title3 = extract_title(title)
    if year!=None :
        sample['year']=year
    else :
        sample['year']=2011
    if ex_title3 == None:
        ex_title3 = province
    sample = find(sample,'ex_title3',ex_title3,col_list)
    sample = find(sample,'variety',variety,col_list)
    sample = find(sample,'winery',winery,col_list)
    sample = find(sample,'taster_name',taster_name,col_list)
    sample = find(sample,'designation',designation,col_list)
    sample = find(sample,'country',country,col_list)
    sample = bow(sample,describe,col_list)
    sa = np.array(sample)
    sa = sa.reshape(1,5233)
    with open('pca_500_model','rb') as f:
        pca = pickle.load(f)  
    data = pd.DataFrame(sa,columns=col_list)
    data = data.drop('price',axis=1)
    data = pca.transform(data)
    
    with open('Normalizer','rb') as f:
        nor = pickle.load(f)
    X = nor.transform(data)
    K.clear_session()
    mlp = keras.models.load_model('mlp.model')
    
    price = mlp.predict(X)
    
    return price[0][0]

if __name__ == '__main__' :
    title='Bee Hunter 2015 Docker Hill Vineyard Pinot Noir (Mendocino)'
    points=96
    describe='The nose of this wine from the folks at Balo Vineyards is full of ripe fruit, giving it a powerful yet nuanced feel. Intense, deep fruit flavor mingles with sage, rhubarb, black tea and a touch of black pepper. It\'s delicious and complex on the palate, with smooth, fine layers of integrated tannins and acidity. Best after 2023.'
    designation = 'Docker Hill Vineyard'
    
    variety='Pinot Noir'
    
    winery='Bee Hunter'
    
    taster_name='Jim Gordon'
    
    country='US'
    
    province='California'
    
    list=[title,points,describe,designation,variety,winery,taster_name,country,province]  
    price = predict(list)
    print(price)
    
