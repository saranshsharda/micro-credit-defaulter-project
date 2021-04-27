#!/usr/bin/env python
# coding: utf-8

# In[86]:


from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[87]:


df = pd.read_csv(r"C:\Users\HP\Desktop\Datafile1.csv",engine='python')


# In[88]:


df.head()


# In[89]:


df.shape


# In[90]:


df.columns


# In[91]:


df.info()


# In[92]:


df.isnull().sum()


# In[93]:


df['label'].value_counts()


# In[94]:


df.drop(['msisdn'], axis = 1,inplace=True)


# In[95]:


df['pcircle'].value_counts()


# In[96]:


df.drop(['pdate'], axis = 1,inplace=True)


# In[97]:


df.drop(['pcircle'], axis = 1,inplace=True)


# In[98]:


df.drop("Unnamed: 0",axis=1,inplace=True)


# In[99]:


df.head()


# In[100]:


plt.figure(figsize=(10,10))
sns.countplot(x="label", data=df)
plt.show()


# In[101]:


countofdefaulter = len(df[df.label == 0])
countofnotdefaulter = len(df[df.label == 1])
print("Percentage of non defaulter: {:.2f}%".format((countofdefaulter / (len(df.label))*100)))
print("Percentage of defaulter: {:.2f}%".format((countofnotdefaulter / (len(df.label))*100)))


# In[102]:


df.describe()


# In[103]:


df.shape


# In[104]:


# scaling part


# In[105]:


df.aon.plot(kind='density')


# In[106]:


#Target Variable (Label)
df_label = df.iloc[:,0]


# In[107]:


#Dropping the target variable from dataframe
df.drop("label",axis=1,inplace=True)


# In[108]:


headnames=[str(i) for i in df.columns]


# In[109]:


from sklearn.preprocessing import Normalizer
scaller=Normalizer()
df = scaller.fit_transform(df)


# In[110]:


type(df)


# In[111]:


df=pd.DataFrame(df,columns=headnames[0:])


# In[112]:


df.shape


# In[113]:


df.head()


# In[114]:


#Concatinating both NUMERIC and CATEGORICAL variables
df = pd.concat([df,df_label], axis=1)
df.head()


# In[115]:


#FEATURE SELECTION


# In[116]:


#Defining the Target and FEATURE Variable
x = df.drop(labels=['label'],axis=1)
y = df.iloc[:,-1]


# In[117]:


x


# In[118]:


y


# In[119]:


from scipy import stats
x[(np.abs(stats.zscore(x))<3).all(axis=1)]


# In[120]:


#Daily amount spent from main account, averaged over last 90 days (in Indonesian Rupiah)
plt.figure(figsize=(10,6))
ax = sns.boxplot(y, x['daily_decr90'])
ax.set_title('Effect of Daily amount spent over last 90 days on Delinquency', fontsize=18)
ax.set_ylabel('Daily amount spent over last 90 days(in Indonesian Rupiah)', fontsize = 15)
ax.set_xlabel('Delinquency', fontsize = 15)


# In[121]:


#Daily amount spent from main account, averaged over last 30 days (in Indonesian Rupiah)
plt.figure(figsize=(10,6))
ax = sns.boxplot(y, x['daily_decr30'])
ax.set_title('Effect of Daily amount spent over last 30 days on Delinquency', fontsize=18)
ax.set_ylabel('Daily amount spent over last 30 days(in Indonesian Rupiah)', fontsize = 15)
ax.set_xlabel('Delinquency', fontsize = 15)


# In[122]:


#Average main account balance over last 90 days
plt.figure(figsize=(10,6))
ax = sns.boxplot(y, x['rental90'])
ax.set_title('Effect of Average balance over last 90 days on Delinquency', fontsize=18)
ax.set_ylabel('Avg balance over last 90 days', fontsize = 15)
ax.set_xlabel('Delinquency', fontsize = 15)


# In[123]:


#Average main account balance over last 30 days
plt.figure(figsize=(10,6))
ax = sns.boxplot(y, x['rental30'])
ax.set_title('Effect of Average balance over last 30 days on Delinquency', fontsize=18)
ax.set_ylabel('Avg balance over last 30 days', fontsize = 15)
ax.set_xlabel('Delinquency', fontsize = 15)


# In[124]:


#train test split


# In[125]:


#create seperate train and test splits for validation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[126]:


#Model selection using various classifiers (Logistic Reg, Random Forest, Decision Tree, Naive Bayes, SVC)


# In[127]:


from sklearn.metrics import f1_score,roc_curve, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform


# In[128]:


def evaluate_model(model):
    model.fit(x_train,y_train)
    prediction_test = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, prediction_test)
    rocauc = metrics.roc_auc_score(y_test, prediction_test)
    return accuracy,rocauc,prediction_test


# In[201]:


#RANDOM FOREST CLASSIFIER
rf =RandomForestClassifier()
rf.fit(x_train,y_train)
acc,rocauc,testpred_rf  = evaluate_model(rf)
print('Random Forest...')
Y_RFpred=rf.predict(x_test)
print(classification_report(Y_RFpred,y_test))


# In[202]:


cm_RF = confusion_matrix(y_test, Y_RFpred)
_,ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm_RF,annot=True,fmt="d")
ax.set_ylim(2,-0.1)


# In[203]:


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
acc,rocauc,testpred_lr  = evaluate_model(lr)
print('Logistic Regression...')
Y_LRpred=lr.predict(x_test)
print(classification_report(Y_LRpred,y_test))


# In[204]:


cm_LR = confusion_matrix(y_test, Y_LRpred)
_,ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm_LR,annot=True,fmt="d")
ax.set_ylim(2,-0.1)


# In[205]:


from sklearn.tree import DecisionTreeClassifier
dt =DecisionTreeClassifier()
dt.fit(x_train,y_train)
acc,rocauc,testpred_dt = evaluate_model(dt)
print('Decision Tree Classifier...')
Y_DTpred=dt.predict(x_test)
print(classification_report(Y_DTpred,y_test))


# In[206]:


cm_DT = confusion_matrix(y_test, Y_DTpred)
_,ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm_DT,annot=True,fmt="d")
ax.set_ylim(2,-0.1)


# In[207]:


#GAUSSIAN NAIVEBAYES
from sklearn.naive_bayes import GaussianNB
gnb =DecisionTreeClassifier()
gnb.fit(x_train,y_train)
acc,rocauc,testpred_gnb = evaluate_model(gnb)
print('Gaussian Navie Bayes...')
Y_GNBpred=gnb.predict(x_test)
print(classification_report(Y_GNBpred,y_test))


# In[208]:


cm_GNB = confusion_matrix(y_test, Y_GNBpred)
_,ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm_GNB,annot=True,fmt="d")
ax.set_ylim(2,-0.1)


# In[209]:


#SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
svc =DecisionTreeClassifier()
svc.fit(x_train,y_train)
acc,rocauc,testpred_svc = evaluate_model(svc)
print('Support vector classifier...')
Y_SVCpred=svc.predict(x_test)
print(classification_report(Y_SVCpred,y_test))


# In[210]:


cm_SVC = confusion_matrix(y_test, Y_SVCpred)
_,ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm_SVC,annot=True,fmt="d")
ax.set_ylim(2,-0.1)


# In[211]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
preds=lr.predict_proba(x)[:,1]
fpr,tpr,thershold=roc_curve(y,preds,drop_intermediate=False)
roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr,'r',label='AUC=%0.2f'% roc_auc)
plt.plot([0,1],[0,1],color='blue',lw=2,linestyle='--')
plt.legend(loc='lower right')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('roc curve')
plt.show()


# In[212]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
preds=rf.predict_proba(x)[:,1]
fpr,tpr,thershold=roc_curve(y,preds,drop_intermediate=False)
roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr,'r',label='AUC=%0.2f'% roc_auc)
plt.plot([0,1],[0,1],color='blue',lw=2,linestyle='--')
plt.legend(loc='lower right')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('roc curve')
plt.show()


# In[213]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
preds=dt.predict_proba(x)[:,1]
fpr,tpr,thershold=roc_curve(y,preds,drop_intermediate=False)
roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr,'r',label='AUC=%0.2f'% roc_auc)
plt.plot([0,1],[0,1],color='blue',lw=2,linestyle='--')
plt.legend(loc='lower right')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('roc curve')
plt.show()


# In[214]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
preds=gnb.predict_proba(x)[:,1]
fpr,tpr,thershold=roc_curve(y,preds,drop_intermediate=False)
roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr,'r',label='AUC=%0.2f'% roc_auc)
plt.plot([0,1],[0,1],color='blue',lw=2,linestyle='--')
plt.legend(loc='lower right')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('roc curve')
plt.show()


# In[215]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
preds=svc.predict_proba(x)[:,1]
fpr,tpr,thershold=roc_curve(y,preds,drop_intermediate=False)
roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr,'r',label='AUC=%0.2f'% roc_auc)
plt.plot([0,1],[0,1],color='blue',lw=2,linestyle='--')
plt.legend(loc='lower right')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('roc curve')
plt.show()


# In[216]:


# Predicted values
y_head_lr = lr.predict(x_test)
y_head_dt = dt.predict(x_test)
y_head_svc = svc.predict(x_test)
y_head_gnb = gnb.predict(x_test)
y_head_rf = rf.predict(x_test)


# In[217]:


from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test,y_head_lr)
cm_dt = confusion_matrix(y_test,y_head_dt)
cm_rf = confusion_matrix(y_test,y_head_rf)
cm_svc = confusion_matrix(y_test,y_head_svc)
cm_gnb = confusion_matrix(y_test,y_head_gnb)


# In[218]:


plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("Randon forest Confusion Matrix")
sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,4)
plt.title("Decision Tree Confusion Matrix")
sns.heatmap(cm_dt,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,5)
plt.title("GAUSSIAN NAIVEBAYES Confusion Matrix")
sns.heatmap(cm_gnb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




