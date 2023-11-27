#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import sklearn
import numpy as n
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


df = pd.read_csv('/Users/manasikakade/Downloads/WA_Fn-UseC_-Telco-Customer-Churn (1).csv')
df.shape


# In[15]:


df.head()


# In[16]:


df.tail()


# In[21]:


df.size


# In[22]:


df.dtypes


# In[23]:


df.columns


# In[25]:


df.info()


# In[26]:


df.isnull().sum()


# In[27]:


df.duplicated().sum()


# In[28]:


df['TotalCharges'].dtype 


# In[29]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors = 'coerce')


# In[30]:


df['TotalCharges'].dtype 


# In[31]:


categorical_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
target = "Churn"


# In[32]:


df.skew(numeric_only= True)


# In[33]:


df.corr(numeric_only= True)


# # Feature Distribution 
# 

# We plot distributions for numerical and categorical features to check for outliers and compare feature distributions with target variable.

# # Numerical features distribution

# Numeric summarizing techniques (mean, standard deviation, etc.) don't show us spikes, shapes of distributions and it is hard to observe outliers with it. That is the reason we use histograms.

# In[34]:


df[numerical_features].describe()


# In[35]:


df[numerical_features].hist(bins=30, figsize=(10, 7))


# We look at distributions of numerical features in relation to the target variable. We can observe that the greater TotalCharges and tenure are, the less is the probability of churn.

# In[36]:


fig, ax = plt.subplots(1, 3, figsize=(14, 4))
df[df.Churn == "No"][numerical_features].hist(bins=30, color="blue", alpha=0.5, ax=ax)
df[df.Churn == "Yes"][numerical_features].hist(bins=30, color="red", alpha=0.5, ax=ax)


# # Categorical feature distribution

# To analyze categorical features, we use bar charts. We observe that Senior citizens and customers without phone service are less represented in the data.

# In[39]:


ROWS, COLS = 4, 4
fig, ax = plt.subplots(ROWS,COLS, figsize=(19,19))
row, col = 0, 0,
for i, categorical_feature in enumerate(categorical_features):
    if col == COLS - 1:
        row += 1
    col = i % COLS
    df[categorical_feature].value_counts().plot(kind='bar', ax=ax[row, col]).set_title(categorical_feature)


# The next step is to look at categorical features in relation to the target variable. We do this only for contract feature. Users who have a month-to-month contract are more likely to churn than users with long term contracts.

# In[40]:


feature = 'Contract'
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df[df.Churn == "No"][feature].value_counts().plot(kind='bar', ax=ax[0]).set_title('not churned')
df[df.Churn == "Yes"][feature].value_counts().plot(kind='bar', ax=ax[1]).set_title('churned')


# # Target variable distribution

# In[42]:


df[target].value_counts().plot(kind='bar').set_title('churned')


# Target variable distribution shows that we are dealing with an imbalanced problem as there are many more non-churned as compare to churned users. The model would achieve high accuracy as it would mostly predict majority class - users who didn't churn in our example.
# 
# Few things we can do to minimize the influence of imbalanced dataset:
# 
# resample data,
# collect more samples,
# use precision and recall as accuracy metrics.

# # OUTLIERS ANALYSIS WITH IQR METHOD 

# In[43]:


x = ['tenure','MonthlyCharges']
def count_outliers(data,col):
        q1 = data[col].quantile(0.25,interpolation='nearest')
        q2 = data[col].quantile(0.5,interpolation='nearest')
        q3 = data[col].quantile(0.75,interpolation='nearest')
        q4 = data[col].quantile(1,interpolation='nearest')
        IQR = q3 -q1
        global LLP
        global ULP
        LLP = q1 - 1.5*IQR
        ULP = q3 + 1.5*IQR
        if data[col].min() > LLP and data[col].max() < ULP:
            print("No outliers in",i)
        else:
            print("There are outliers in",i)
            x = data[data[col]<LLP][col].size
            y = data[data[col]>ULP][col].size
            a.append(i)
            print('Count of outliers are:',x+y)
global a
a = []
for i in x:
    count_outliers(df,i)


# # Cleaning and Transforming Data

# In[44]:


df.drop(['customerID'],axis = 1,inplace = True)


# In[45]:


df.head()


# Dropped customerID because it is not needed

# # One Hot Encoding

# In[47]:


df1=pd.get_dummies(data=df,columns=['gender', 'Partner', 'Dependents', 
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'], drop_first=True)


# # df1.head()

# In[49]:


df1.columns


# # Rearranging Columns

# In[51]:


df1 = df1[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'gender_Male', 'Partner_Yes', 'Dependents_Yes',
       'PhoneService_Yes', 'MultipleLines_No phone service',
       'MultipleLines_Yes', 'InternetService_Fiber optic',
       'InternetService_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No internet service',
       'OnlineBackup_Yes', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No internet service',
       'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No internet service', 'StreamingMovies_Yes',
       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check','Churn_Yes']]


# In[52]:


df1.head()


# In[53]:


df1.shape


# In[54]:


from sklearn.impute import SimpleImputer

# The imputer will replace missing values with the mean of the non-missing values for the respective columns

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

df1.TotalCharges = imputer.fit_transform(df1["TotalCharges"].values.reshape(-1, 1))


# # Feature Scaling

# In[55]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[56]:


scaler.fit(df1.drop(['Churn_Yes'],axis = 1))
scaled_features = scaler.transform(df1.drop('Churn_Yes',axis = 1))


# In[57]:


from sklearn.model_selection import train_test_split
X = scaled_features
Y = df1['Churn_Yes']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=44)


# # Prediction using Logistic Regression

# In[58]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score ,confusion_matrix

logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)


# In[59]:


predLR = logmodel.predict(X_test)


# In[60]:


predLR


# In[61]:


Y_test


# In[62]:


print(classification_report(Y_test, predLR))


# In[63]:


# calculate the classification report
report = classification_report(Y_test, predLR, target_names=['Churn_No', 'Churn_Yes'])

# split the report into lines
lines = report.split('\n')

# split each line into parts
parts = [line.split() for line in lines[2:-5]]

# extract the metrics for each class
class_metrics = dict()
for part in parts:
    class_metrics[part[0]] = {'precision': float(part[1]), 'recall': float(part[2]), 'f1-score': float(part[3]), 'support': int(part[4])}

# create a bar chart for each metric
fig, ax = plt.subplots(1, 4, figsize=(12, 4))
metrics = ['precision', 'recall', 'f1-score', 'support']
for i, metric in enumerate(metrics):
    ax[i].bar(class_metrics.keys(), [class_metrics[key][metric] for key in class_metrics.keys()])
    ax[i].set_title(metric)

# display the plot
plt.show()


# In[64]:


confusion_matrix_LR = confusion_matrix(Y_test, predLR)


# In[65]:


# create a heatmap of the matrix using matshow()

plt.matshow(confusion_matrix(Y_test, predLR))

# add labels for the x and y axes
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix_LR[i, j], ha='center', va='center')


# Add custom labels for x and y ticks
plt.xticks([0, 1], ["Not Churned", "Churned"])
plt.yticks([0, 1], ["Not Churned", "Churned"])
plt.show()


# In[66]:


logmodel.score(X_train, Y_train)


# In[67]:


accuracy_score(Y_test, predLR)


# # Prediction using Support Vector Classifier

# In[68]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, Y_train)
y_pred_svc = svc.predict(X_test)


# In[69]:


print(classification_report(Y_test, y_pred_svc))


# In[70]:


confusion_matrix_svc = confusion_matrix(Y_test, y_pred_svc)


# In[71]:


# create a heatmap of the matrix using matshow()

plt.matshow(confusion_matrix_svc)

# add labels for the x and y axes
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix_svc[i, j], ha='center', va='center')

        
# Add custom labels for x and y ticks
plt.xticks([0, 1], ["Not Churned", "Churned"])
plt.yticks([0, 1], ["Not Churned", "Churned"])
plt.show()


# In[72]:


svc.score(X_train,Y_train)


# In[73]:


accuracy_score(Y_test, y_pred_svc)


# # Prediction using Decision Tree Classifier

# In[74]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train, Y_train)
y_pred_dtc = dtc.predict(X_test)


# In[75]:


print(classification_report(Y_test, y_pred_dtc))


# In[76]:


confusion_matrix_dtc = confusion_matrix(Y_test, y_pred_dtc)


# In[77]:


# create a heatmap of the matrix using matshow()

plt.matshow(confusion_matrix_dtc)

# add labels for the x and y axes
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix_dtc[i, j], ha='center', va='center')


# Add custom labels for x and y ticks
plt.xticks([0, 1], ["Not Churned", "Churned"])
plt.yticks([0, 1], ["Not Churned", "Churned"])
plt.show()


# In[78]:


dtc.score(X_train,Y_train)


# In[79]:


accuracy_score(Y_test, y_pred_dtc)


# # Prediction using KNN Classifier

# In[82]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 30)
knn.fit(X_train,Y_train)


# In[83]:


pred_knn = knn.predict(X_test)


# In[84]:


error_rate= []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,Y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != Y_test))


# In[85]:


plt.figure(figsize = (10,6))
plt.plot(range(1,40),error_rate,color = 'blue',linestyle = '--',marker = 'o',markerfacecolor='red',markersize = 10)
plt.title('Error Rate vs K')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[86]:


print(classification_report(Y_test,pred_knn))


# In[87]:


confusion_matrix_knn = confusion_matrix(Y_test,pred_knn)


# In[88]:


# create a heatmap of the matrix using matshow()

plt.matshow(confusion_matrix_knn)

# add labels for the x and y axes
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix_knn[i, j], ha='center', va='center')

# Add custom labels for x and y ticks
plt.xticks([0, 1], ["Not Churned", "Churned"])
plt.yticks([0, 1], ["Not Churned", "Churned"])
plt.show()


# In[89]:


knn.score(X_train,Y_train)


# In[90]:


accuracy_score(Y_test, pred_knn)


# In[ ]:




