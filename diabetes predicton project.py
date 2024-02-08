#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[7]:


diabete_datasets = pd.read_csv('diabetes.csv')


# In[18]:


diabete_datasets.head(10)


# In[17]:


diabete_datasets.head(-10)


# In[20]:


diabete_datasets.shape


# In[ ]:





# In[25]:


diabete_datasets.describe()


# In[30]:


diabete_datasets['Outcome'].value_counts()


# In[33]:


### 0 mean non diabetes patient , 1 mean diabetes patient


# In[42]:


diabete_datasets['Outcome'].mean()


# In[45]:


diabete_datasets.groupby('Outcome').mean()


# In[46]:


# saperate data and label
X = diabete_datasets.drop(columns = 'Outcome', axis =1)
Y = diabete_datasets['Outcome']


# In[47]:


print(X)


# In[48]:


print(Y)


# In[49]:


# data standerzation


# In[56]:


scaler = StandardScaler()


# In[59]:


scaler.fit(X)


# In[62]:


standardized_data = scaler.transform(X)


# In[63]:


print(standardized_data)


# In[68]:


X = standardized_data
Y = diabetes_datasets['Outcome']


# In[69]:


print(X)
print(Y)


# In[70]:


# train test split


# In[71]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[72]:


print(X.shape, X_train.shape, X_test.shape)


# In[73]:


classifier = svm.SVC(kernel='linear')


# In[74]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[75]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[76]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[77]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[78]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[79]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




