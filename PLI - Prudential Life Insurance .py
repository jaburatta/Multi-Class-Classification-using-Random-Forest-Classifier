#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# import necessary Libraries
import numpy as np 
import pandas as pd 
from sklearn.feature_selection import RFE, VarianceThreshold,RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score,RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,auc,roc_curve,cohen_kappa_score
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns',30)


# In[ ]:


# Read in the training dataset
train_data = pd.read_csv("C:/Users/USER/Desktop/Capstone/prudential_train.csv")
print(train_data.shape)
print(train_data.info())


# In[ ]:


# Read in the test dataset
test_data = pd.read_csv("C:/Users/USER/Desktop/Capstone/prudential_test.csv")
test_data.shape


# In[ ]:


# List of training features with missing data 
train_null_feat = train_data.columns[train_data.isnull().sum()> 0]
print(train_null_feat, "\n" "There are " + str(len(train_null_feat)) + " training features with missing data")


# In[ ]:


# Training Features with >10% missing values from total observation
highnuldata = train_data.columns[train_data.isnull().sum()> 0.10*train_data.shape[0]]
print(highnuldata, "\n" + str(len(highnuldata)) + " training features have greater than 10% missing data")


# In[ ]:


# List of test data features with missing data 
test_null_feat = test_data.columns[test_data.isnull().sum()> 0]
print(test_null_feat, "\n" "There are " + str(len(test_null_feat)) + " test features with missing data")


# In[ ]:


# Test Features with >10% missing values from total observation
highnulldata = test_data.columns[test_data.isnull().sum()> 0.10*test_data.shape[0]]
print(highnulldata,  "\n" + str(len(highnulldata)) + " test dataset features have greater than 10% missing data")


# In[ ]:


# drop features with missign values greater than 10% of total observations 
new_train_data= train_data.drop(train_data[highnuldata], axis=1)
new_test_data= test_data.drop(test_data[highnulldata], axis=1)

print(new_train_data.shape)
print(new_test_data.shape)


# In[ ]:


# we impute data into features with missing value <= 10%
new_train_data = new_train_data.fillna(new_train_data.mean())


# In[ ]:


new_test_data = new_test_data.fillna(new_test_data.mean())


# # Feature Engineering

# In[ ]:


# a function to add encoded features together
def add_cols(df):
    cols = df.columns
    for i in range(len(cols)):
        cols_total= df[cols].sum(axis=1)
    
    return cols_total


# In[ ]:


# creating a new medical keyword feature which is a sum of the total medical keyword for each instance
medicalkw = new_train_data.loc[ :, new_train_data.columns.str.startswith ("Medical_Keyword_")]
new_train_data['Medical_Keyword'] = add_cols(medicalkw)


# In[ ]:


medicalkw = new_test_data.loc[ :, new_test_data.columns.str.startswith ("Medical_Keyword_")]
new_test_data['Medical_Keyword'] = add_cols(medicalkw)


# In[ ]:


# creating a new medical history feature which is a sum of the total medical keyword for each instance
medicalHist= new_train_data.loc[ :, new_train_data.columns.str.startswith ("Medical_History_")]
new_train_data['Medical_History'] = add_cols(medicalHist)


# In[ ]:


medicalHist= new_test_data.loc[ :, new_test_data.columns.str.startswith ("Medical_History_")]
new_test_data['Medical_History'] = add_cols(medicalHist)


# In[ ]:


# A product of BMI and Ins_Age
new_train_data['BMI_InsAge']=new_train_data['BMI']*new_train_data['Ins_Age']


# In[ ]:


new_test_data['BMI_InsAge']=new_test_data['BMI']*new_test_data['Ins_Age']


# In[ ]:


# Drop some features to prevent redundancy 
new_train_data=new_train_data.drop(['BMI','Ins_Age',"Id",'Ht','Wt'],axis=1)


# In[45]:


new_train_data=new_train_data.drop(medicalkw,axis=1)


# In[46]:


new_train_data=new_train_data.drop(medicalHist,axis=1)


# In[47]:


new_test_data=new_test_data.drop(['BMI','Ins_Age','Ht','Wt'],axis=1)


# In[48]:


new_test_data=new_test_data.drop(medicalkw,axis=1)


# In[49]:


new_test_data=new_test_data.drop(medicalHist,axis=1)


# In[ ]:


# Check for correlation in the training dataset
def get_correlated(data, threshold):
    cols = set()
    corrmat = data.corr()
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if corrmat.iloc[i,j] > threshold:
                colname = corrmat.columns[i]
                cols.add(colname)
    return cols


# In[ ]:


# correlating features to drop 
to_drop = get_correlated(new_train_data, 0.90)
to_drop


# In[ ]:


new_train_data= new_train_data.drop(to_drop, axis=1)
new_train_data.shape


# In[ ]:


test_to_drop = get_correlated(new_test_data, 0.90)
test_to_drop


# In[ ]:


new_test_data= new_test_data.drop(test_to_drop, axis=1)


# In[ ]:


# make a copy of applicant id from test data
Id = new_test_data['Id'].copy()


# In[ ]:


new_test_data  = new_test_data.drop('Id', axis = 1)


# In[ ]:


# save column names from dataset and unique values in product info 2 which will serve as column later
cols=list(X.columns.copy())
prod_info_cols = list(X['Product_Info_2'].unique())
X_cols=prod_info_cols + cols
item = 'Product_Info_2'
index = X_cols.index(item)
del X_cols[index]
len(X_cols)


# In[ ]:


# Our Model won't understand alphanumeric input so We label encode product info 2 feature
# column_trans handles different preprocessing steps for different features,
# and pass through the remaining features that does not require any preprocessiong step

column_trans = make_column_transformer(
    (OneHotEncoder(), ['Product_Info_2']),
    remainder = StandardScaler())


# In[ ]:


X = column_trans.fit_transform(X)
print(X.shape)


# In[ ]:


new_test_data = column_trans.transform(new_test_data)
new_test_data


# In[ ]:


# Balancing our training dataset using resampling by over-sampling
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=27)
X, y = sm.fit_sample(X, y)
X.shape, y.shape
                                                                     
y.value_counts()


# In[ ]:


# make X a dataframe again and use column names saved earlier X_cols
X = pd.DataFrame(X,columns=X_cols)


# In[ ]:


# Check for zero variance or degenerate variables and drop them
near_zero = VarianceThreshold(threshold = 0.10)
near_zero.fit_transform(X)
highvar = near_zero.get_support(indices=True)

print(X.columns[highvar], len(X.columns[highvar]))


# In[ ]:


X = X[X.columns[highvar]]


# In[ ]:


# make new_test_data dataframe again and use column names saved earlier X_cols
new_test_data = pd.DataFrame(new_test_data,columns=X_cols)
new_test_data= new_test_data[new_test_data.columns[highvar]]


# # Cross Validation 

# In[ ]:


# Cross Validation using Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100,n_jobs=-1)
rfc_scores=cross_val_score(rfc, X,y, cv=10, scoring='accuracy')
print(rfc_scores)
print(rfc_scores.mean())


# In[ ]:


# Cross Validation using K Neighbors CLassifier
knn = KNeighborsClassifier()
knn_scores = cross_val_score(knn, X,y, cv=10, scoring='accuracy')
print(knn_scores)
print(knn_scores.mean())


# # HyperParameter Tuning using Randomized search CV

# In[ ]:


# K Neighbors Classifier

neighbors = list(range(5,35,5))
param_dist = dict(n_neighbors=neighbors)

knn_random = RandomizedSearchCV(knn, param_dist,cv=10, scoring='accuracy',n_iter=10,random_state=5, n_jobs=-1) 
knn_random.fit(X,y)


# In[ ]:


print('K Neighbor Classifier Best Accuracy: ',knn_random.best_score_)
print('K Neighbor Classifier Best tuned parameter: ',knn_random.best_params_)
print('K Neighbor Classifier Best tuned parameters: ', knn_random.best_estimator_)


# In[ ]:


# Random Forest Classifier 

param_dist = dict(n_estimators = [100,200])
rfc_random = RandomizedSearchCV(rfc, param_dist,cv=10, scoring='accuracy',n_iter=10,random_state=5, n_jobs=-1) 
rfc_random.fit(X,y)


# In[ ]:


print('Random Forest Classifier Best Accuracy: ',rfc_random.best_score_)
print('Random Forest Classifier Best tuned parameter: ',rfc_random.best_params_)
print('Random Forest Classifier Best tuned parameters: ',rfc_random.best_estimator_)


# # Modelling

# In[ ]:


# Using the tuned parameters on Random Forest 
clf=RandomForestClassifier(n_estimators=200,n_jobs=-1)
clf.fit(X,y)


# In[ ]:


# Predicitng some samples from dataset it was trained with
eval_pred = clf.predict(X_eval[highvar])


# In[ ]:


# The result from predicted samples and how they compare to the actual
eval_result = pd.DataFrame({'y_train_actual': y_eval,'y_train_predicted':eval_pred})
eval_result.head()


# In[ ]:


# Predicting Response for the test dataset
test_pred = clf.predict(new_test_data)


# In[ ]:


# Applicant Id that was copied earlier is now used to track response 
test_data_result = pd.DataFrame({'Applicant Id':Id ,'Predicted Response':test_pred})
test_data_result.head()


# In[ ]:


# Count of test_data instances to Predicted Response
test_data_result['Predicted Response'].value_counts()


# In[ ]:


# save the trained model into disk using pickle
pickle.dump(clf, open('model.pkl','wb'))


# In[ ]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# In[ ]:


print(model.predict([[0,0,1,26,0.23,2,3,1,0.027,9,1,2,1,2,6,3,1,2,1,2,1,1,3,2,2,12,156,0.34]]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




