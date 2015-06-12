"""
West nile prediction
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
from sklearn import cross_validation as CV
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import math

# Load dataset

train_data = pd.read_csv('../input/train.csv')
#train_data = pd.read_csv('trainZeroFilled.csv',delimiter="|")
test_data = pd.read_csv('../input/test.csv')
sample_data = pd.read_csv('../input/sampleSubmission.csv')
weather_data = pd.read_csv('../input/weather.csv')

# Confirm data 
"""
print train_data
print test_data
print sample_data
print weather_data.data
"""

# Get Labels

labels = train_data.WnvPresent.values

# Drop features

weather_data = weather_data.drop('CodeSum', axis=1)

# Split station 1 and 2 and join horizontally
weather_stn1 = weather_data[weather_data['Station']==1]
weather_stn2 = weather_data[weather_data['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather_data = weather_stn1.merge(weather_stn2, on='Date')


# replace some missing values and T with -1
weather_data= weather_data.replace('M', -1)
weather_data= weather_data.replace('-', -1)
weather_data= weather_data.replace('T', -1)
weather_data= weather_data.replace(' T', -1)
weather_data= weather_data.replace('  T', -1)

# Functions to extract month and date from the dataset
def create_month(x):
    return x.split('-')[1]

def create_day(x):
    return x.split('-')[2]

def create_year(x):
    return x.split('-')[0]

train_data['month'] = train_data.Date.apply(create_month)
train_data['day'] = train_data.Date.apply(create_day)
test_data['month'] = test_data.Date.apply(create_month)
test_data['day'] = test_data.Date.apply(create_day)
train_data['year'] = train_data.Date.apply(create_year)
test_data['year'] = test_data.Date.apply(create_year)

train_data['monthYear'] = train_data.month.map(str) + train_data.year.map(str)
test_data['monthYear'] = test_data.month.map(str) + test_data.month.map(str)


# Add integer latitude/longitude columns
train_data['Lat_int'] = train_data.Latitude.apply(int)
train_data['Long_int'] = train_data.Longitude.apply(int)
test_data['Lat_int'] = test_data.Latitude.apply(int)
test_data['Long_int'] = test_data.Longitude.apply(int)


train_data = train_data.drop(['Address', 'AddressNumberAndStreet','WnvPresent', 'NumMosquitos'], axis = 1)
test_data = test_data.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)


# Merge with weather data
train_data = train_data.merge(weather_data, on='Date')
test_data = test_data.merge(weather_data, on='Date')
train_data = train_data.drop(['Date'], axis = 1)
test_data = test_data.drop(['Date'], axis = 1)

# Add satellite feature identifier
train_data['satelliteFeature'] = train_data.Trap.str[4:]
test_data['satelliteFeature'] = test_data.Trap.str[4:]

train_data.to_csv('test.csv', index=False)


pattern = r'[A-Z]'
train_data['isSatelliteFeature'] = train_data.satelliteFeature.str.contains(pattern) 
test_data['isSatelliteFeature'] = test_data.satelliteFeature.str.contains(pattern) 

patternA = r'A'
train_data['isSatelliteFeatureA'] = train_data.satelliteFeature.str.contains(patternA)
test_data['isSatelliteFeatureA'] = test_data.satelliteFeature.str.contains(patternA)

patternB = r'B'
train_data['isSatelliteFeatureB'] = train_data.satelliteFeature.str.contains(patternB)
test_data['isSatelliteFeatureB'] = test_data.satelliteFeature.str.contains(patternB)

patternC = r'C'
train_data['isSatelliteFeatureC'] = train_data.satelliteFeature.str.contains(patternC)
test_data['isSatelliteFeatureC'] = test_data.satelliteFeature.str.contains(patternC)

# Convert categorical data to numbers
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train_data['Species'].values) + list(test_data['Species'].values))
train_data['Species'] = lbl.transform(train_data['Species'].values)
test_data['Species'] = lbl.transform(test_data['Species'].values)

lbl.fit(list(train_data['Street'].values) + list(test_data['Street'].values))
train_data['Street'] = lbl.transform(train_data['Street'].values)
test_data['Street'] = lbl.transform(test_data['Street'].values)

lbl.fit(list(train_data['Trap'].values) + list(test_data['Trap'].values))
train_data['Trap'] = lbl.transform(train_data['Trap'].values)
test_data['Trap'] = lbl.transform(test_data['Trap'].values)

lbl.fit(list(train_data['satelliteFeature'].values) + list(test_data['satelliteFeature'].values))
train_data['satelliteFeature'] = lbl.transform(train_data['satelliteFeature'].values)
test_data['satelliteFeature'] = lbl.transform(test_data['satelliteFeature'].values)

lbl.fit(list(train_data['isSatelliteFeature'].values) + list(test_data['isSatelliteFeature'].values))
train_data['isSatelliteFeature'] = lbl.transform(train_data['isSatelliteFeature'].values)
test_data['isSatelliteFeature'] = lbl.transform(test_data['isSatelliteFeature'].values)

lbl.fit(list(train_data['isSatelliteFeatureA'].values) + list(test_data['isSatelliteFeatureA'].values))
train_data['isSatelliteFeatureA'] = lbl.transform(train_data['isSatelliteFeatureA'].values)
test_data['isSatelliteFeatureA'] = lbl.transform(test_data['isSatelliteFeatureA'].values)

lbl.fit(list(train_data['isSatelliteFeatureB'].values) + list(test_data['isSatelliteFeatureB'].values))
train_data['isSatelliteFeatureB'] = lbl.transform(train_data['isSatelliteFeatureB'].values)
test_data['isSatelliteFeatureB'] = lbl.transform(test_data['isSatelliteFeatureB'].values)

lbl.fit(list(train_data['isSatelliteFeatureC'].values) + list(test_data['isSatelliteFeatureC'].values))
train_data['isSatelliteFeatureC'] = lbl.transform(train_data['isSatelliteFeatureC'].values)
test_data['isSatelliteFeatureC'] = lbl.transform(test_data['isSatelliteFeatureC'].values)

# Add point distances from keypoints

train_dist_pt1 = pow(pow(train_data['Latitude'].values - 41.974689,2) + pow(train_data['Longitude'].values + 87.890615,2),2) * 100
train_dist_pt2 = pow(pow(train_data['Latitude'].values - 41.878114,2) + pow(train_data['Longitude'].values + 87.629798,2),2) * 100
train_dist_pt3 = pow(pow(train_data['Latitude'].values - 41.673408,2) + pow(train_data['Longitude'].values + 87.599862,2),2) * 100

test_dist_pt1 = pow(pow(test_data['Latitude'].values - 41.974689,2) + pow(test_data['Longitude'].values + 87.890615,2),2) * 100
test_dist_pt2 = pow(pow(test_data['Latitude'].values - 41.878114,2) + pow(test_data['Longitude'].values + 87.629798,2),2) * 100
test_dist_pt3 = pow(pow(test_data['Latitude'].values - 41.673408,2) + pow(test_data['Longitude'].values + 87.599862,2),2) * 100

train_data['dist_pt1'] = train_dist_pt1
test_data['dist_pt1'] = test_dist_pt1

train_data['dist_pt2'] = train_dist_pt2
test_data['dist_pt2'] = test_dist_pt2

train_data['dist_pt3'] = train_dist_pt3
test_data['dist_pt3'] = test_dist_pt3

train_data.to_csv('tmp.csv', index=False)

# drop columns with -1s
train_data = train_data.ix[:,(train_data != -1).any(axis=0)]
test_data = test_data.ix[:,(test_data != -1).any(axis=0)]

# drop columns with low importance
train_data =  train_data.drop(['WetBulb_y', 'Tavg_y', 'ResultSpeed_x', 'WetBulb_x', 'AvgSpeed_x', 'Tmin_y', 'AvgSpeed_y', 'ResultDir_y', 'ResultDir_x', 'SnowFall_x', 'Long_int', 'Depth_x'], axis = 1)
test_data = test_data.drop(['WetBulb_y', 'Tavg_y', 'ResultSpeed_x', 'WetBulb_x', 'AvgSpeed_x', 'Tmin_y', 'AvgSpeed_y', 'ResultDir_y', 'ResultDir_x', 'SnowFall_x', 'Long_int', 'Depth_x'], axis = 1)

train_data, labels = shuffle(train_data, labels, random_state=123)
clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=1, max_depth=30)
#clf = ensemble.RandomForestClassifier()
clf.fit(train_data, labels)

#Backfitting
backfitting = clf.predict_proba(train_data)[:,1]


# create predictions and submission file
predictions = clf.predict_proba(test_data)[:,1]
sample_data['WnvPresent'] = predictions
sample_data.to_csv('randomForestClassifier_addedFeatures_depth30.csv', index=False)

# Cross-validation
#train_data_tail = train_data.tail(4000)
#labels_data_tail = pd.DataFrame(labels)
#labels_data_tail = labels_data_tail.tail(4000)

#predictions_tail = clf.predict_proba(train_data_tail)[:,1]

#print mean_squared_error(labels_data_tail, predictions_tail)

print 'Printing Backfitting ROC FPR and TPR'
fpr, tpr, thresholds = roc_curve(labels, backfitting)
print fpr
print tpr

print 'Printing Backfitting ROC Accuracy score'
accuracy_score = roc_auc_score(labels, backfitting)
print accuracy_score

print 'Cross Validation Score'
print CV.cross_val_score(clf, train_data, labels, cv=5) 
