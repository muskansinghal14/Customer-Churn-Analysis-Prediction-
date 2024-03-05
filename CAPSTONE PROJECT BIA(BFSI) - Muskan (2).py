#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore') 


# In[6]:


df = pd.read_csv(r"C:\Users\ajayg\Desktop\2.Bank_Cust_Churn_Modelling.csv", encoding= "unicode_escape") 
df 


# In[3]:


df.head() 


# In[4]:


df.tail() 


# ### Exploratory Data Analysis (EDA) 

# In[5]:


df.describe() 


# In[6]:


df.info() 


# In[7]:


df.isnull() 


# In[8]:


df.isnull().sum() 


# In[9]:


df.nunique() 


# In[10]:


df.duplicated() 


# In[12]:


df.duplicated().sum() 


# In[13]:


df.ndim


# In[14]:


df.shape


# In[15]:


df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True) 


# In[16]:


df.shape


# # Data Visualization 

# In[17]:


labels = 'Exited', 'Retained'
sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.show()


# **INSIGHT- So about 20% of the customers have churned and around 79% have retained. So the baseline model could be to predict that 20% of the customers will churn. Given 20% is a small number, we need to ensure that the chosen model does predict with great accuracy this 20% as it is of interest to the bank to identify and keep this bunch as opposed to accurately predicting the customers that are retained.**

# In[18]:


fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='Geography', hue = 'Exited',data = df, ax=axarr[0][0])
sns.countplot(x='Gender', hue = 'Exited',data = df, ax=axarr[0][1])
sns.countplot(x='HasCrCard', hue = 'Exited',data = df, ax=axarr[1][0])
sns.countplot(x='IsActiveMember', hue = 'Exited',data = df, ax=axarr[1][1]) 
plt.show() 


# **INSIGHT :- 
# (1).Majority of the data is from persons from France where majority is towards customers who are retained.
# (2.)The proportion of female customers churning is also greater than that of male customers
# (3.)Interestingly, majority of the customers that churned are those with credit cards. Given that majority of the customers have credit cards could prove this to be just a coincidence.
# (4.)Unsurprisingly the inactive members have a greater churn. Worryingly is that the overall proportion of inactive mebers is quite high suggesting that the bank may need a program implemented to turn this group to active customers as this will definately have a positive impact on the customer churn.**

# In[19]:


fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = df, ax=axarr[0][0])
sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = df , ax=axarr[0][1])
sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][0])
sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][1])
sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][1])
plt.show() 


# **INSIGHT :- 1)There is no significant difference in the credit score distribution between retained and churned customers.
# 2)The older customers are churning at more than the younger ones alluding to a difference in service preference in the age categories. The bank may need to review their target market or review the strategy for retention between the different age groups
# 3)With regard to the tenure, the clients on either extreme end (spent little time with the bank or a lot of time with the bank) are more likely to churn compared to those that are of average tenure.
# 4)Worryingly, the bank is losing customers with significant bank balances which is likely to hit their available capital for lending.
# 5)Neither the product nor the salary has a significant effect on the likelihood to churn.** 

# In[72]:


numerical_features = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_features.corr()
plt.figure(figsize=(10, 8))

sns.heatmap(correlation_matrix, annot=True, cmap='crest', fmt='.2f', linewidths=0.5)

plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
plt.show() 


# **INSIGHT 
# "Exited" has a positive correlation with "Age" as it has the largest positive correlation value among all columns.
# "Exited" has a weaker positive correlation with "Balance" & "EstimatedSalary" columns.
# "Exited" has a negative correlation with "IsActiveMember" & "NumofProducts" columns.
# "Exited" has no correlation with "Tenure" column.**

# In[21]:


numerical_features = df.select_dtypes(include=['float64', 'int64'])

sns.set(style='whitegrid')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='CreditScore', y='Age', hue='Exited', data=df, palette={0: 'blue', 1: 'green'}, alpha=0.5)
plt.title('Age vs CreditScore (Hue: Exited)', fontsize=16)
plt.xlabel('CreditScore', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.legend(title='Exited', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# **INSIGHT - Insight that can be taken from the scatterplot of CreditScore VS Age with the points color coded by whether customers Exited or not shows that the majority of customers who exited were between 40-60 years old and the majority of customers who does not exited were between 20-50 years old.**

# ### FEATURE ENGINEERING 

# In[22]:


exit_counts = df['Exited'].value_counts()
print(exit_counts)


# *We have imbalanced dataset, there are fewer records on customers who have churned (Exited=1) compared to those who haven't (Exited=0)*

# In[23]:


churn_percentage = df.groupby('HasCrCard')['Exited'].mean() * 100
churn_percentage


# *Having a credit card (HasCrCard) doesn't seem to influence the likelihood of churn* 

# In[24]:


df['Geography'].unique() 


# In[25]:


df['Gender'].unique()


# In[26]:


gender =  pd.get_dummies(df['Gender'], drop_first = True)
Geography = pd.get_dummies(df['Geography'], drop_first=True) 


# In[27]:


df.drop(['Gender', 'Geography', 'HasCrCard'], axis=1, inplace=True) 


# In[63]:


df = pd.concat([df, gender, Geography], axis=1) 
df 


# ### Train Test Split 

# In[3]:


#Using Over-Sampling technique for my imbalanced data 
from imblearn.over_sampling import RandomOverSampler 
from sklearn.model_selection import train_test_split


# In[7]:


ros = RandomOverSampler(sampling_strategy=1)
X = df.drop(['Exited'], axis=1)
y = df['Exited']
X, y = ros.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) 


# In[31]:


X.shape


# In[32]:


y.shape


# In[33]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ### MODELS (Assessing Predictive Accuracy) 

# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 


# ### *1) LOGISTIC REGRESSION* 

# In[44]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train) 


# In[45]:


y_pred = logmodel.predict(X_test)
y_pred 


# In[46]:


print("Classification Report:") 
print(classification_report(y_test, y_pred)) 

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred)) 


# In[47]:


Logistic_Regression_Accuracy = accuracy_score(y_test, y_pred) 
print("Logistic_Regression_Accuracy is :{:.2f}%".format(Logistic_Regression_Accuracy*100)) 


# ### *2) DECISION TREES*  

# In[48]:


from sklearn.tree import DecisionTreeClassifier 


# In[49]:


clf = DecisionTreeClassifier() 
clf.fit(X_train, y_train) 

predictions_dt = clf.predict(X_test)  


# In[50]:


print("Classification Report:") 
print(classification_report(y_test, predictions_dt))  

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions_dt))  


# In[51]:


Decision_Tree_Accuracy = accuracy_score(y_test, predictions_dt) 
print("Decision_Tree_Accuracy is :{:.2f}%".format(Decision_Tree_Accuracy*100)) 


# ### *3) RANDOM FOREST CLASSIFIER* 

# In[52]:


from sklearn.ensemble import RandomForestClassifier 


# In[53]:


model = RandomForestClassifier() 
model.fit(X_train, y_train) 

y_pred_rf = model.predict(X_test) 


# In[54]:


print("Classification Report:") 
print(classification_report(y_test, y_pred_rf))  

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))   


# In[60]:


RandomForest_Accuracy = accuracy_score(y_test, y_pred_rf) 
print("RandomForest_Accuracy is :{:.2f}%".format(RandomForest_Accuracy*100))   


# ### *4) GRADIENT BOOSTING* 

# In[56]:


from sklearn.ensemble import GradientBoostingClassifier


# In[58]:


model_gb = GradientBoostingClassifier() 
model_gb.fit(X_train, y_train) 

y_pred_gb = model_gb.predict(X_test)  


# In[59]:


print("Classification Report:") 
print(classification_report(y_test, y_pred_gb))  

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))  


# In[62]:


accuracy_gb = accuracy_score(y_test, y_pred_gb) 
print("Gradient Boosting Accuracy: {:.2f}%".format(accuracy_gb*100))   


# ### Compare all 4 Models

# In[64]:


# Define accuracy values
accuracy_values = [Logistic_Regression_Accuracy * 100, Decision_Tree_Accuracy*100, RandomForest_Accuracy*100, accuracy_gb*100]

# Create a DataFrame for model comparison
compare = pd.DataFrame({'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
 'Accuracy': accuracy_values})

# Sort the DataFrame by Accuracy in descending order
compare = compare.sort_values(by='Accuracy', ascending=False)

# Print the comparison DataFrame
print(compare) 


# ### MODEL COMPARISION VISUAL 

# In[73]:


# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(compare['Model'], compare['Accuracy'], color='blue')
plt.title('*Model Comparison*',color = 'red')
plt.xlabel('Model',color ='purple')
plt.ylabel('Accuracy (%)',color = 'purple')
plt.ylim(0, 100) # Set the y-axis limit from 0 to 100 for percentage
plt.show() 


# ### CONCLUSION - *In all the four models, Random Forest has achieved the highest accuracy which is 93.63%.* 

# In[ ]:




