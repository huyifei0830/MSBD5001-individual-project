# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:28:57 2019

@author: huyifei
"""

import pandas as pd
from sklearn.decomposition import PCA 
#from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
#read train.csv
data=pd.read_csv('E:/课/上/5001/msbd5001-fall2019/train.csv')
#data processing
data.dropna(inplace=True)
data=data.reset_index()
data['purchase_date'] = pd.to_datetime(data['purchase_date'])
data['release_date'] = pd.to_datetime(data['release_date'])
data['pur_year'] = data['purchase_date'].apply(lambda dt: dt.year)
data['pur_month'] = data['purchase_date'].apply(lambda dt: dt.month)
data['pur_day'] = data['purchase_date'].apply(lambda dt: dt.day)
data['re_year'] = data['release_date'].apply(lambda dt: dt.year)
data['re_month'] = data['release_date'].apply(lambda dt: dt.month)
data['re_day'] = data['release_date'].apply(lambda dt: dt.day)
genres = data['genres'].str.get_dummies(",")#one-hot encoding
categories = data['categories'].str.get_dummies(",")
tags = data['tags'].str.get_dummies(",")
pca=PCA(n_components=3)     #Load PCA algorithm, set the number of the principal Components to 3 after dimension reduction
reduced_genres=pca.fit_transform(genres)#dimension reduction
reduced_categories=pca.fit_transform(categories)
reduced_tags=pca.fit_transform(tags)
data['categories0'],data['categories1'],data['categories2']=reduced_categories[:,0],reduced_categories[:,1],reduced_categories[:,2]
data['tags0'],data['tags1'],data['tags2']=reduced_tags[:,0],reduced_tags[:,1],reduced_tags[:,2]
X=data.drop(['purchase_date', 'release_date', 'genres', 'categories','tags','playtime_forever','index','id'], axis=1)
y=data['playtime_forever']
"""
#use grid search to find the best parameters
X_train = X
y_train = y 
gbr = GradientBoostingRegressor(random_state=0)
param_grid = {
    'n_estimators': [100,200,300],
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    'learning_rate': [0.1],
    'subsample': [1]
}
model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=5)
model.fit(X_train, y_train)
print('Gradient boosted tree regression...')
print('Best Params:')
print(model.best_params_)
print('Best CV Score:')
print(-model.best_score_)
"""
#bulid and train the model
model_gbr = GradientBoostingRegressor(max_depth=4)
model_gbr.fit(X, y)
#read test.csv
data_test=pd.read_csv('E:/课/上/5001/msbd5001-fall2019/test.csv')
data_test.fillna(0,inplace=True)
data_test=data_test.reset_index()
data_test['purchase_date'] = pd.to_datetime(data_test['purchase_date'])
data_test['release_date'] = pd.to_datetime(data_test['release_date'])
data_test['pur_year'] = data_test['purchase_date'].apply(lambda dt: dt.year)
data_test['pur_month'] = data_test['purchase_date'].apply(lambda dt: dt.month)
data_test['pur_day'] = data_test['purchase_date'].apply(lambda dt: dt.day)
data_test['re_year'] = data_test['release_date'].apply(lambda dt: dt.year)
data_test['re_month'] = data_test['release_date'].apply(lambda dt: dt.month)
data_test['re_day'] = data_test['release_date'].apply(lambda dt: dt.day)
genres_test = data_test['genres'].str.get_dummies(",")
categories_test = data_test['categories'].str.get_dummies(",")
tags_test = data_test['tags'].str.get_dummies(",")
pca=PCA(n_components=3)
reduced_genres_test=pca.fit_transform(genres_test)
reduced_categories_test=pca.fit_transform(categories_test)
reduced_tags_test=pca.fit_transform(tags_test)
data_test['categories0'],data_test['categories1'],data_test['categories2']=reduced_categories_test[:,0],reduced_categories_test[:,1],reduced_categories_test[:,2]
data_test['tags0'],data_test['tags1'],data_test['tags2']=reduced_tags_test[:,0],reduced_tags_test[:,1],reduced_tags_test[:,2]
X_test=data_test.drop(['purchase_date', 'release_date', 'genres', 'categories','tags','index','id'], axis=1)
y_test = model_gbr.predict(X_test)
#we want to predict the playtime, and the absolute value of negative numbers in y_test are small, so we set them to 0
for i in range(len(y_test)):
    if y_test[i]<0:
        y_test[i]=0
#get pred_sub.csv for submission
submission = pd.read_csv('E:/课/上/5001/msbd5001-fall2019/samplesubmission.csv')
sub = pd.DataFrame(data=y_test, index=submission.id)
sub.reset_index(inplace=True)
sub.columns = submission.columns
sub.to_csv('E:/课/上/5001/msbd5001-fall2019/pred_sub_4.csv', index=False)