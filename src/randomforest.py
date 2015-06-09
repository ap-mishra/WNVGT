"""
random forest
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

# Load dataset 
#train = pd.read_csv('../input/trainZeroFilled.csv',delimiter="|")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sampleSubmission.csv')
weather = pd.read_csv('../input/weather.csv')

print "data load complete"

# Get labels
labels = train.WnvPresent.values
print "labels generated"
print labels
print labels.shape

# Not using codesum for this benchmark
weather = weather.drop('CodeSum', axis=1)

# Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

# replace some missing values and T with -1
weather = weather.replace('M', -1)
weather = weather.replace('-', -1)
weather = weather.replace('T', -1)
weather = weather.replace(' T', -1)
weather = weather.replace('  T', -1)

# Functions to extract month and day from dataset
# You can also use parse_dates of Pandas.
def create_month(x):
    return x.split('-')[1]

def create_day(x):
    return x.split('-')[2]
print 'Stage 1'
print train.head()

train['month'] = train.Date.apply(create_month)
train['day'] = train.Date.apply(create_day)
test['month'] = test.Date.apply(create_month)
test['day'] = test.Date.apply(create_day)

print 'Stage 2: Adding Month and day'
print train.head()

# Add integer latitude/longitude columns
train['Lat_int'] = train.Latitude.apply(int)
train['Long_int'] = train.Longitude.apply(int)
test['Lat_int'] = test.Latitude.apply(int)
test['Long_int'] = test.Longitude.apply(int)

# drop address columns
train = train.drop(['WnvPresent', 'NumMosquitos'], axis = 1)
test = test.drop(['Id'], axis = 1)

print 'Stage 3: Dropping address and other features'
print train.head()

# Merge with weather data
train = train.merge(weather, on='Date')
test = test.merge(weather, on='Date')
train = train.drop(['Date'], axis = 1)
test = test.drop(['Date'], axis = 1)

print 'Stage 4: Merging weather data'
print train.head()
print "Final shape of training data set"
print train.shape

# Convert categorical data to numbers
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['Species'].values) + list(test['Species'].values))
train['Species'] = lbl.transform(train['Species'].values)
test['Species'] = lbl.transform(test['Species'].values)

lbl.fit(list(train['Street'].values) + list(test['Street'].values))
train['Street'] = lbl.transform(train['Street'].values)
test['Street'] = lbl.transform(test['Street'].values)

lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
train['Trap'] = lbl.transform(train['Trap'].values)
test['Trap'] = lbl.transform(test['Trap'].values)

#'Address', 'AddressNumberAndStreet'
lbl.fit(list(train['Address'].values) + list(test['Address'].values))
train['Address'] = lbl.transform(train['Address'].values)
test['Address'] = lbl.transform(test['Address'].values)

lbl.fit(list(train['AddressNumberAndStreet'].values) + list(test['AddressNumberAndStreet'].values))
train['AddressNumberAndStreet'] = lbl.transform(train['AddressNumberAndStreet'].values)
test['AddressNumberAndStreet'] = lbl.transform(test['AddressNumberAndStreet'].values)

print 'Stage 5: After Categorical -> Numerical transition using LabelEncoder'
print train.head()
print train.shape

print 'Also : here is the test set preview'
print test.head()
print test.shape

# drop columns with -1s
train = train.ix[:,(train != -1).any(axis=0)]
test = test.ix[:,(test != -1).any(axis=0)]

print 'Dropping some columns completely for missing information. What is my new shape?'
print train.shape
print test.shape

possible_species = train['Species'].unique()

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs=-1, n_estimators=1000, min_samples_split=1)

model.fit(train, labels)

# Backfitting - without proba
backfitting = model.predict_proba(train)[:,1]

print 'Shape of backfitted data'
print backfitting

# create predictions and submission file
predictions = model.predict_proba(test)[:,1]
sample['WnvPresent'] = predictions
sample['WnvPresent'] = sample.WnvPresent.apply(float)

sample.to_csv('random_forest.csv', index=False)

from sklearn import cross_validation as CV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import svm

print 'Mean square error and cross val score'
print labels.shape
print predictions.shape
print mean_squared_error(labels, backfitting)

print 'Printing Backfitting ROC FPR and TPR'
fpr, tpr, thresholds = roc_curve(labels, backfitting)
print fpr
print tpr

#import matplotlib.pyplot as plt

#plt.plot(fpr,tpr)
#plt.show()

print 'Printing Backfitting ROC Accuracy score'
accuracy_score = roc_auc_score(labels, backfitting)
print accuracy_score


print 'Printing confusion matrix.'
from sklearn.metrics import confusion_matrix
c_matrix = confusion_matrix(labels, backfitting)

print c_matrix

#clf = svm.SVC(kernel='linear', C=1)
#print CV.cross_val_score(clf, train, labels, cv=5)
