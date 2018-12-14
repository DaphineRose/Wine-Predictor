

# Wine Price Predictor
Final Project : INFO 7390 Advances In Data Science

<img src="img\1fc1b9c372cff55d.png" alt="img\1fc1b9c372cff55d.png"  width="259.00" />

Members:
Wenqi Cui,
Yuchen Qiao,
Byron Kiriibwa


Advisor:
Sri Krishnamurthy


TA： Pramod Nagare


## Topic Overview



When customers selecting wine, they will face many problems. How does winery effect price of wine? Does the rating of the wine provided by the wine enthusiast have a bearing on the price of the wine? And what other factors decide price of wine? This application will give prediction of price according to user's input.

This application can  provide prediction of price for customers when they input informations of wine.


## Data Source



[https://www.kaggle.com/zynicide/wine-reviews](https://www.kaggle.com/zynicide/wine-reviews)

The data was scraped from Wine Enthusiast during the week of June 15th, 2017.( [http://www.winemag.com/?s=&drink_type=wine](http://www.winemag.com/?s=&drink_type=wine) ) 


## Pipeline Design



#### Data cleaning


The original dataset has 13 columns. Original data information:


|Unnamed:0|country|description|designation|points|price|province|region_1|region_2|taster_name|taster_twitter_handle|title|variety|winery|
|------ | ------ | ------ |------ | ------ | ------ |------ | ------ | ------ |------ | ------ | ------ |------ | ------ |
|0|Italy|Aromas include tropical fruit, broom, brimston...|Vulkà Bianco|87|NaN|Sicily & Sardinia|Etna|NaN|Kerin O’Keefe|@kerinokeefe|  Nicosia 2013 Vulkà Bianco (Etna)|White Blend|Nicosia|
|1|Portugal|This is ripe and fruity, a wine that is smooth...  |Avidagos  |87  |15.0|  Douro|  NaN|  NaN|  Roger Voss|  @vossroger|Quinta dos Avidagos 2011 Avidagos Red (Douro)  |Portuguese Red|  Quinta dos Avidagos|



There are lots of missing values, we treat them in different ways.


|  |column_name| missing_count|
|------ | ------ | ------ |
|0| country | 59|
|1|  description|  0|
|2 | designation | 34779|
|3 | points| 0|
|4 | price | 8996|
|5 | province | 59|
|6 | region_1 | 19575|
|7  |region_2 |70683|
|8 | taster_name | 24496|
|9 | taster_twitter_handle  |29416|
|10 | title | 0|
|11 | variety | 1|
|12 | winery | 0|

For the target column('price'), we just drop out the row.

```
data=origin_data.dropna(axis=0,subset=['price'])
```

For numerical columns, we fill null by mean.

```python
data['year'].fillna(2011,inplace=True)
data['points'].fillna(88,inplace=True)
```

For categorical columns, we will keep the null value, make NA a new level.


We noticed that this dataset contains several columns about location so that it is possible to use other columns' information to fill missing values and finally combine those relative columns. Region_1 and Region_2 columns describe detail area where the wine are produced and we can use information from title to replace missing value. Here is the code to split title column.

```
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
```

Then, we combine columns relate to area into one column ex_title3:

```
 tmp=0
for i in ii:
    tmp = data.loc[i,'region_1']
    if tmp is not np.nan:
        data.loc[i,'ex_title3']=tmp
    else:
        tmp = data.loc[i,'province']
        if tmp is not np.nan:
            data.loc[i,'ex_title3']=tmp   
```



#### Exploratory Data Analysis

<img src="img\6460012b1fdb0b4e.png" alt="img\6460012b1fdb0b4e.png"  width="519.50" />

New column-produced year of wine

<img src="img\6f3a6663a09a86de.png" alt="img\6f3a6663a09a86de.png"  width="536.62" />

Country percentage(top 20 countries)

<img src="img\ba52456cb4674067.png" alt="img\ba52456cb4674067.png"  width="528.50" />

Percentage of points

<img src="img\761726cf506269bc.png" alt="img\761726cf506269bc.png"  width="558.50" />

Relation between points and price

<img src="img\3628a1d1e585e071.png" alt="img\3628a1d1e585e071.png"  width="602.00" />

Relation between country and price

<img src="img\4311411085134ea2.png" alt="img\4311411085134ea2.png"  width="602.00" />

Relation between country and price

<img src="img\d142938165bbed5a.png" alt="img\d142938165bbed5a.png"  width="602.00" />

Average points for every country

#### Feature Engineering

There are huge amount of category columns in this dataset. 

In title column, it contains multiple information so we split it to several new columns.  

Title columns example:

<img src="img\5d645449edb924de.png" alt="img\5d645449edb924de.png"  width="1819.13" />

For description column, we convert it to a BOW. 

> DRich and juicy, this youthful, structured wine is from one of Volnay's top vineyards. It is heading towards ripe opulence and dense black-cherry flavors as it matures. Concentration and fruit characterize this impressive wine. Drink from 2023.


First, we split all descriptions in to independent words and import nltk package to drop all stop-words. Then, we calculate frequency of each words after which we select top 200 listed words.

The result like follow:

{',': 0,
 '.': 1,
 'thi': 2,
 'wine': 3,

......
 'doe': 197,
 'petit': 198,
 'tangi': 199}
 
Word cloud of description column:

<img src="img\a7d52a295f2a5b36.png" alt="img\a7d52a295f2a5b36.png"  width="701.38" />

Word cloud(frequency of description column)

For other category columns, we use one hot encoding to convert them into numeric columns. However, there are too many unique values in some columns and converting all of them will resulting in enlargement of columns.

|country|  description | designation | points |  province|  region_1|  region_2 | taster_name | taster_twitter_handle | title | variety | winery|
|------ | ------ | ------ |------ | ------ | ------ |------ | ------ | ------ |------ |------ |------ | 
|1 | 42 | 111567 | 35776 | 21  | 422  |1204 | 17 | 19 | 15 | 110638| 697  |15855|

To solve this problem, we decide to select certain values that cover most percentage of the range(like 90%), for the long tails, we convert them to a new value 'other'. Then, we convert those values into numeric type.  We create a function to_dummy to do this automatically.

Here is code for this function:

```
def to_dummy(data_copy,col,cut):
    n = data_copy[col].nunique()
    m = data_copy.shape[0] * cut
    counts = data_copy[col].value_counts()
    su = 0
    ot=[]
    for i in range(n):
        su += counts[i]
        if su>m:
            ot = counts.index[i+1:]
            break
    fil=[x in ot for x in data_copy[col]]
    data_copy[col][fil]='other'        

    dummy = pd.get_dummies(data_copy[col],dummy_na=True,prefix=col)
    data_copy=data_copy.join(dummy)
    data_copy=data_copy.drop(col,axis=1)  
    gc.collect()
    return data_copy
```

 Here is how data look like after converting:

<img src="img\a6a56374d9d11b85.png" alt="img\a6a56374d9d11b85.png"  width="926.86" />

After we process all columns, the dataframe become too large for future analysis(over 5000 columns). To reduce the number of columns without causing loss to prediction, we use PCA and decrease number of columns to 500.

PCA introduction:

Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.

Code for using PCA in this application:

```
pca = PCA(n_components=500)
pca_X = pca.fit_transform(X)
pca_X = pd.DataFrame(pca_X)
pca_X.shape
```

Date shape after using PCA:

(120975, 500)

#### Model Selection, Training and  Evaluation

In model selection part, we test follow models: Decision Tree Regression, Random Forest Regression and Gradient Boosting Regression and NLP. After comparing result several models, we select MLP as our model.

Here are result of some models:

Random Forest Regression:
```
rf_reg = RandomForestRegressor(max_depth=50,max_features=100,n_estimators=30,verbose=2)
```

```
Score: 0.42920096985536543
Score for Train: 0.8365994461786193
RMS: 29.050122262155245
MAPE: 42.119624733119124
R2: 0.4292009698553655
MAE: 12.795948103605777
```
<img src="img\41bcc0b5724b140b.png" alt="img\41bcc0b5724b140b.png"  width="602.00" />

Gradient Boosting Regression:

```
gb_reg = GradientBoostingRegressor(max_depth=20,max_features=20,n_estimators=15,verbose=2,learning_rate=0.1)
```

```
Score: 0.33843386322790747
Score for Train: 0.9133864178492134
```
 


Keras introduction:

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Keras sequential(Using 4 layers) code: 

```
model = Sequential()

model.add(Dense(500,activation='relu',input_shape=X.shape[1:]))
model.add(BatchNormalization())
model.add(Dropout(rate=dropout_rate))

model.add(Dense(500,activation='relu',input_shape=X.shape[1:]))
model.add(BatchNormalization())
model.add(Dropout(rate=dropout_rate))

model.add(Dense(100,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=dropout_rate))

model.add(Dense(20,activation='relu'))
model.add(Dense(1))
model.summary()  
```

Output:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   

=================================================================
dense_21 (Dense)             (None, 500)               250500    
_________________________________________________________________
batch_normalization_11 (Batc (None, 500)               2000      
_________________________________________________________________
dropout_11 (Dropout)         (None, 500)               0         
_________________________________________________________________
dense_22 (Dense)             (None, 500)               250500    
_________________________________________________________________
batch_normalization_12 (Batc (None, 500)               2000      
_________________________________________________________________
dropout_12 (Dropout)         (None, 500)               0         
_________________________________________________________________
dense_23 (Dense)             (None, 100)               50100     
_________________________________________________________________
batch_normalization_13 (Batc (None, 100)               400       
_________________________________________________________________
dropout_13 (Dropout)         (None, 100)               0         
_________________________________________________________________
dense_24 (Dense)             (None, 20)                2020      
_________________________________________________________________
dense_25 (Dense)             (None, 1)                 21        

=================================================================
Total params: 557,541
Trainable params: 555,341
Non-trainable params: 2,200
_________________________________________________________________

```
optimizer = optimizers.adam(lr=learning_rate,decay=decay)
model.compile(loss='mse',
              optimizer=optimizer, 
              metrics=['mae','mse'])
history=model.fit(X, Y,
          batch_size=256,
          epochs=50,
          verbose=1,
          callbacks=cb,
          validation_split=0.3,
          shuffle=True)
```


<img src="img\55e3068d4a8d096a.png" alt="img\55e3068d4a8d096a.png"  width="493.00" />

<img src="img\29f3a39786f4761d.png" alt="img\29f3a39786f4761d.png"  width="504.00" />

<img src="img\25875f7d5e3f8787.png" alt="img\25875f7d5e3f8787.png"  width="504.00" />

####  Model Deployment 

We use flask to design a web application and deploy it on AWS.


## Details on how to run the model



We used a combination of Flask and AWS to be able to deploy the wine price predictor model and made predictions through a website. 

#### 1.The Prediction

In order to be able to make predictions, users were expected to enter the title, points, describe, designation, variety, winery, taster_name, country and province.

<img src="img\ef9e06469208f92f.png" alt="img\ef9e06469208f92f.png"  width="414.50" />

On pressing the predict button, users are routed to the results page that shows them the predicted price of the wine that they put into the prediction page.

<img src="img\167b6fb1f0e72dee.png" alt="img\167b6fb1f0e72dee.png"  width="602.00" />

#### 2. User input

When the user inputs information on the front page and presses predict, the data is first stored as list;

<img src="img\20655de89bf83c74.png" alt="img\20655de89bf83c74.png"  width="602.00" />

The list is passed to a function predict_mod which preprocesses data, converts it into a format that can be used by the model to be able to make predictions. Before predictions, different parts of the input are preprocessed differently as shown below;

#### 3. Website design

**Title: **The title is sliced into different four different parts, from which the wine year is extracted using the function below;

<img src="img\88640c37a74a8f60.png" alt="img\88640c37a74a8f60.png"  width="602.00" />

After the preprocessing, we call the predict function using the newly input data to predict the price of wine.

The prediction is made using an already trained model, that is loaded using pickle.

To make the prediction, we import important libraries and use two flask templates to display the results.

<img src="img\c80a2e9fa3f8aa2f.png" alt="img\c80a2e9fa3f8aa2f.png"  width="602.00" />

Flask uses a combination of two templates;

**Home.html**

The home.html is used to collect data from the user, here is the structure of the home template.

<img src="img\d8064045cd771388.png" alt="img\d8064045cd771388.png"  width="602.00" />

Once the user presses the predict button, they are re-routed to the results page where the results are displayed

<img src="img\27f0019f27a9b21.png" alt="img\27f0019f27a9b21.png"  width="602.00" />

The html templates are saved into a templates folder, while the app.py file that links the templates is stored in the same folder.

**Amazon Web Services**

We created an AWS instance and we installed all the necessary software

<img src="img\9aaef50cae8f5524.png" alt="img\9aaef50cae8f5524.png"  width="602.00" />

All the files and the model to be uploaded to AWS are stored in the same folder.

<img src="img\669e6a5f87d607ef.png" alt="img\669e6a5f87d607ef.png"  width="602.00" />

We use scp to upload all the files to the AWS instance

<img src="img\74bc8c21c8c98540.png" alt="img\74bc8c21c8c98540.png"  width="602.00" />

We run the sudo python app.py command to start the application.

<img src="img\810674cac641f1f2.png" alt="img\810674cac641f1f2.png"  width="602.00" />

Users can be able to access the app and to get predictions on the link below;

http://18.191.144.104/


## Reference



[1].  [https://www.kaggle.com/zynicide/wine-reviews/home](https://www.kaggle.com/zynicide/wine-reviews/home) 

[2].  [https://www.winemag.com/?s=&drink_type=wine](https://www.winemag.com/?s=&drink_type=wine) 

[3].  [https://www.datacamp.com/community/tutorials/wordcloud-python](https://www.datacamp.com/community/tutorials/wordcloud-python) 

[4].  [https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html) 

[5].  [https://www.nltk.org/](https://www.nltk.org/) 

[6].  [https://scikit-learn.org/stable/supervised_learning.html#supervised-learning](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning) 

[7]. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) 

[8].  [https://en.wikipedia.org/wiki/Principal_component_analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) 

[9].  [http://setosa.io/ev/principal-component-analysis/](http://setosa.io/ev/principal-component-analysis/) 

[10].  [https://keras.io/](https://keras.io/)

