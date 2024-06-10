#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""1. Data Exploration:
a. Load the dataset and perform exploratory data analysis (EDA).
b. Examine the features, their types, and summary statistics.
c. Create visualizations such as histograms, box plots, or pair plots
to visualize the distributions and relationships between features.
Analyze any patterns or correlations observed in the data.
"""

#import the files
import pandas as pd
df_train  = pd.read_csv("D:/DATA SCIENCE EXCELR/Data/Titanic_train.csv")
df_test  = pd.read_csv("D:/DATA SCIENCE EXCELR/Data/Titanic_test.csv")


# In[2]:


#EDA
df_train.shape


# In[3]:


df_train.head()


# In[4]:


df_train.info()


# In[5]:


df_test.info()


# In[6]:


#Data Cleaning
#filling the missing values with mean in the train data
df_train["Age"]=df_train["Age"].fillna(df_train["Age"].mean().round(1))
df_train.info()


# In[7]:


#filling the missing values with mean in the test data 
df_test["Age"]=df_test["Age"].fillna(df_test["Age"].mean().round(1))
df_test.info()


# In[8]:


#removing unneccesary columns in train data
df_train.drop(["Survived","PassengerId","Name","Cabin","Ticket"],axis=1,inplace=True)
df_train.info()


# In[9]:


#removing unneccesary columns in test data
df_test.drop(["PassengerId","Name","Cabin","Ticket"],axis=1,inplace=True)
df_test.info()


# In[10]:


#dropping null values in train data
df_train.dropna(inplace=True)
df_train.info()


# In[11]:


#dropping null values in test data
df_test.dropna(inplace=True)
df_test.info()


# In[12]:


#Data Transformation

# label encoding the categorical variables in the train data
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df_train['Sex'] = LE.fit_transform(df_train['Sex'])
df_train['Embarked'] = LE.fit_transform(df_train['Embarked'])
df_train


# In[13]:


# label encoding the categorical variables in the test data
df_test['Sex'] = LE.fit_transform(df_test['Sex'])
df_test['Embarked'] = LE.fit_transform(df_test['Embarked'])
df_test


# In[14]:


#standardizing the continous variables in the train data
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
df_train["Age"] = SS.fit_transform(df_train[["Age"]])
df_train["Fare"] = SS.fit_transform(df_train[["Fare"]])
df_train["SibSp"] = SS.fit_transform(df_train[["SibSp"]])
df_train["Parch"] = SS.fit_transform(df_train[["Parch"]])
df_train.head()


# In[15]:


#standardizing the continous variables in the test data
df_test["Age"] = SS.fit_transform(df_test[["Age"]])
df_test["Fare"] = SS.fit_transform(df_test[["Fare"]])
df_test["SibSp"] = SS.fit_transform(df_test[["SibSp"]])
df_test["Parch"] = SS.fit_transform(df_test[["Parch"]])
df_test.head()


# In[16]:


#defining Y variable in the train variable
Y_train=df_train["Sex"]
df_train.drop("Sex",axis=1,inplace=True) #dropping the Y variable
df_train


# In[17]:


#defining the Y variable in the test data
Y_test=df_test["Sex"]
df_test.drop("Sex",axis=1,inplace=True) #dropping the Y variable 
df_test


# In[18]:


X_train=df_train.iloc[:,0:6] #Defining the x variables in the train data


# In[19]:


X_test=df_test.iloc[:,0:6] #Defining the x variable in the test data


# In[20]:


# step5: Model fitting with the training data
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)


# In[21]:


# predictions with training and testing data
Y_train_pred = logreg.predict(X_train)
Y_test_pred = logreg.predict(X_test)
Y_train_pred # training data predections


# In[22]:


Y_test_pred #testing data predictions


# In[23]:


# step6: Metrics

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
cm1 = confusion_matrix(Y_train,Y_train_pred) #confusion matrix for training data
cm1


# In[24]:


cm2 = confusion_matrix(Y_test,Y_test_pred) #confusion matrix for testing data
cm2


# In[25]:


ac1 = accuracy_score(Y_train,Y_train_pred) #accuracy for the training data
print("Accuracy score:", ac1.round(2))


# In[26]:


ac2 = accuracy_score(Y_test,Y_test_pred) #accuracy for the testing data
print("Accuracy score:", ac2.round(2))


# In[27]:


rs1 = recall_score(Y_train,Y_train_pred) #recall or sensitivity for the training data
print("sensitivity score:", rs1.round(2))


# In[28]:


rs2 = recall_score(Y_test,Y_test_pred) #recall or sensitivity for the testing data
print("sensitivity score:", rs2.round(2))


# In[29]:


TN = cm1[0,0]
FP = cm1[0,1]
TNR = TN/(TN + FP)
print("specificity score:", TNR.round(2)) #Specificity score for the training data


# In[30]:


TN = cm2[0,0]
FP = cm2[0,1]
TNR = TN/(TN + FP)
print("specificity score:", TNR.round(2)) #Specificity for the testing data


# In[31]:


ps1 = precision_score(Y_train,Y_train_pred) #precision for the training data
print("Precision score:", ps1.round(2)) 


# In[32]:


ps2 = precision_score(Y_test,Y_test_pred) #precision for the testing data
print("Precision score:", ps2.round(2))


# In[33]:


f1_1 = f1_score(Y_train,Y_train_pred) #f1 score for the training data
print("F1 score:", f1_1.round(2))


# In[34]:


f1_2 = f1_score(Y_test,Y_test_pred) #f1 score for the testing data
print("F1 score:", f1_2.round(2))


# In[ ]:


"""Based on the metrics we can determine that this model is best suited  in contexts
where correctly identifying positive cases is more important than correctly identifying negative cases.
"""


# In[35]:


from sklearn.metrics import roc_curve,roc_auc_score

df_test["Y_proba"] = logreg.predict_proba(X_test)[:,1] #predicting the probability values of each row 
df_test.head()

fpr,tpr,dummy = roc_curve(Y_test,df_test["Y_proba"]) #roc_curve


# In[36]:


import matplotlib.pyplot as plt
plt.scatter(fpr,tpr)
plt.plot(fpr,tpr,color='red')
plt.xlabel("False positive Rate")
plt.ylabel("True positive Rate")
plt.show()

print("AUC score:", roc_auc_score(Y_test,df_test["Y_proba"]).round(3)) #AUC score


# In[37]:


logreg.coef_


# In[38]:


df_train.info()


# In[ ]:


# Based the logistic regression coefficients, 
"""We can determine that 
The coefficient for 'Pclass' is 0.31808643, indicating that a one-unit increase in Pclass
is associated with an increase of 0.31808643 in the log-odds of the outcome of the target variable.

Similarly For the  'Age' with coefficient of 0.20782688 and 'Embarked' with coefficient of 0.24917129 
impacts that much of the log-odds of the outcome Positively.

But when it comes to SibSp,Parch,Fare with respective negative coefficients
of -0.02168479, -0.46812261, -0.1077161 indicating that a one-unit increase in the features
SibSp,Parch,Farebis associated with the decrease of 0.02168479, 0.46812261, 0.1077161 
in the log-odds of the outcome of the target variable.


SO WE CAN CONCLUDE THAT THE FEATURES Pclass,Age,Embarked HAVE THE POSITIVE IMPACT IN THE LOG-ODDS OF
THE TARGET VARIABLE  WHEREAS SibSp,Parch,Fare HAVE THE NEGATIVE IMPACT IN THE 
LOG-ODDS OF THE TARGET VARIABLE."""

