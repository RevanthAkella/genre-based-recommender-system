
# coding: utf-8

# In[208]:


import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# In[216]:


train_df = pd.read_csv('train.csv',index_col='PassengerId')
test_df = pd.read_csv('test.csv',index_col='PassengerId')
Survived = train_df['Survived'].copy()
train_df = train_df.drop('Survived', axis=1).copy()
df = pd.concat([train_df,test_df])
traindex = train_df.index
testdex = test_df.index
df['FamilySize'] = df['SibSp'] + df['Parch'] +1
df['Name_length'] = df['Name'].apply(len)
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1


# In[212]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


# In[213]:


dtree = DecisionTreeClassifier()


# In[205]:


data = data.drop(["Name","Ticket","Cabin","Embarked","Fare"], axis = 1)
data.replace(to_replace = "male", value = "1", inplace = True)
data.replace(to_replace = "female", value = "0", inplace = True)
data = data.astype(int)


# In[193]:


from sklearn.model_selection import train_test_split


# In[194]:


X = data.drop(["Survived"], axis = 1)


# In[195]:


X


# In[196]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.33, random_state = 42)


# In[197]:


dtree.fit(X_train, Y_train)
y_pred = dtree.predict(X_test)


# In[198]:


from sklearn.metrics import accuracy_score


# In[199]:


print(accuracy_score(y_pred, Y_test))

