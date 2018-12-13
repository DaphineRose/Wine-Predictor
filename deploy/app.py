from flask import Flask, render_template, url_for, request, redirect
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
import pickle
from sklearn.externals import joblib
from pickle import dump 
from pickle import load
import predict_mod

app = Flask(__name__)
@app.route('/')
def index():
	return render_template('home.html')
	
    
#Use None to file the blank of the list
	
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
	    title = request.form['title']
	    points =  request.form['points']
	    describe = request.form['description']
	    designation = request.form['designation']
	    variety =  request.form['variety']
	    winery = request.form['winery']
	    taster_name =  request.form['taster_name']
	    country = request.form['country']
	    province =  request.form['province']
	    list=[title,points,describe,designation,variety,winery,taster_name,country,province]
		

	    my_prediction = predict_mod.predict(list)
		
    return render_template('results.html', prediction = my_prediction)
	#return render_template('results.html')
	
if __name__ == "__main__":
	app.run(debug=True)