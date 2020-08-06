#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import math
from scipy.stats import f_oneway
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
import datetime
import warnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.metrics import r2_score,mean_squared_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
import statsmodels.stats.outliers_influence
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
import pydotplus as pdot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
warnings.filterwarnings('ignore')
from sklearn.model_selection import LeaveOneOut


# In[2]:


os.chdir('D:\project')
df=pd.read_csv('credit-card-data.csv')


# In[3]:


df.info()


# In[4]:


plt.hist(df['MINIMUM_PAYMENTS'],bins='auto')


# In[5]:


df['MINIMUM_PAYMENTS']=df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean())


# In[6]:


df['status']=np.where(df['PAYMENTS']>df['MINIMUM_PAYMENTS'],'not stressed','stressed')


# In[7]:


df['status'].value_counts()


# In[8]:


df_risk=df[df['status']=='not stressed']


# In[9]:


df_risk['Average_Utilization']=(df_risk['BALANCE']/df_risk['CREDIT_LIMIT'])*100
df_risk['Average_Utilization'].describe()


# In[10]:


df_cleaned=df_risk[df_risk['Average_Utilization']<100]
df_cleaned.info()


# In[11]:


df_cleaned['limit_per_trnx']=df_cleaned['PURCHASES_TRX']/df_cleaned['CREDIT_LIMIT']
df_cleaned['limit_per_trnx'].describe()


# In[12]:


df_cleaned['ONEOFF_PURCHASES_FREQUENCY_MONTHS']=np.round((df_cleaned['ONEOFF_PURCHASES_FREQUENCY']*12),0)
df_cleaned['ONEOFF_PURCHASES_FREQUENCY_MONTHS'].describe()


# In[13]:


df_cleaned['PURCHASES_INSTALLMENTS_FREQUENCY_MONTHS']=np.round((df_cleaned['PURCHASES_INSTALLMENTS_FREQUENCY']*12),0)
df_cleaned['PURCHASES_INSTALLMENTS_FREQUENCY_MONTHS'].describe()


# In[14]:


df_cleaned['limit_per_cash_trnx']=df_cleaned['CASH_ADVANCE_TRX']/df_cleaned['CREDIT_LIMIT']
df_cleaned['limit_per_cash_trnx'].describe()


# In[15]:


df_cleaned['Utilization_category']=pd.cut(df_cleaned.Average_Utilization,bins=[0,10,20,30,50,70,2000],labels=['verylow','low','fair','high','veryhigh','risky'],include_lowest=True)
df_cleaned['Utilization_category'].value_counts()


# In[16]:


df_cleaned['ability']=pd.cut(df_cleaned.PRC_FULL_PAYMENT,bins=[0,0.25,0.50,0.75,1],labels=['low','medium','high','very high'],include_lowest=True)
df_cleaned['ability'].value_counts()


# In[17]:


df_cleaned['credit_discipline']=np.where(df_cleaned['Average_Utilization']>30,'not disciplined','disciplined')
df_cleaned['credit_discipline'].value_counts()


# In[18]:


df_cleaned['Frequency_oneoff']=pd.cut(df_cleaned['ONEOFF_PURCHASES_FREQUENCY_MONTHS'],bins=[0,1,5,7,9,13],labels=['verylow','low','medium','high','very high'],include_lowest=True)
df_cleaned['Frequency_oneoff'].value_counts()


# In[19]:


df_cleaned['Frequency_installment']=pd.cut(df_cleaned['PURCHASES_INSTALLMENTS_FREQUENCY_MONTHS'],bins=[0,1,5,7,9,13],labels=['verylow','low','medium','high','veryhigh'],include_lowest=True)
df_cleaned['Frequency_installment'].value_counts()


# In[20]:


df_cleaned['PURCHASES_FREQUENCY'].describe()


# In[21]:


df_cleaned_freq=df_cleaned[df_cleaned['PURCHASES_FREQUENCY']>0.80]
df_cleaned_ideal=df_cleaned_freq[df_cleaned_freq['credit_discipline']=='disciplined']
df_cleaned_ideal['target']=1
df_cleaned_copy=df_cleaned.copy()
ideal=list(df_cleaned_ideal.index)
df_cleaned_target=df_cleaned_copy.drop(ideal,axis=0)
df_cleaned_target=df_cleaned_target.append(df_cleaned_ideal,sort=False)
df_cleaned_target['target']=df_cleaned_target['target'].fillna(0)
df_cleaned_target['target'].value_counts()


# In[22]:


df_status=df['status'].value_counts().reset_index()
sns.barplot(x='status',y='index',data=df_status)
df_status


# In[23]:


df_uti=df_cleaned['Utilization_category'].value_counts().reset_index()
df_uti['pct']=np.round((df_uti['Utilization_category']/df_uti['Utilization_category'].sum())*100,2)
sns.barplot(y='index',x='pct',data=df_uti)
df_uti


# In[24]:


df_ability=df_cleaned['ability'].value_counts().reset_index()
df_ability['pct']=np.round((df_ability['ability']/df_ability['ability'].sum())*100,2)
df_ability


# In[25]:


sns.barplot(x='pct',y='index',data=df_ability)


# In[26]:


pd.crosstab(df_cleaned['ability'],df_cleaned['Utilization_category'])


# In[27]:


pd.crosstab(df_cleaned['ability'],df_cleaned['credit_discipline'])


# In[28]:


df_disc=df_cleaned[df_cleaned ['credit_discipline']=='disciplined']
df_target_customer=df_disc[df_disc['PRC_FULL_PAYMENT']>0.4999]


# In[29]:


df_oneoff=df_target_customer['Frequency_oneoff'].value_counts().reset_index()
df_oneoff['pct']=np.round((df_oneoff['Frequency_oneoff']/df_oneoff['Frequency_oneoff'].sum())*100,2)
sns.barplot(x='pct',y='index',data=df_oneoff)
df_oneoff


# In[30]:


df_installment=df_target_customer['Frequency_installment'].value_counts().reset_index()
df_installment['pct_installment']=np.round((df_installment['Frequency_installment']/df_installment['Frequency_installment'].sum())*100,2)
sns.barplot(x='pct_installment',y='index',data=df_installment)
df_installment


# In[31]:


cnames=['BALANCE','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','Average_Utilization',
                                       'CASH_ADVANCE','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','TENURE']


# In[32]:


df_scaler=df_cleaned.copy()


# In[33]:


for i in cnames:
    df_scaler[i]=(df_scaler[i]-min(df_scaler[i]))/(max(df_scaler[i])-min(df_scaler[i]))


# In[34]:


cluster_range=range(1,10)
cluster_errors=[]
for num_clusters in cluster_range:
    clusters=KMeans(num_clusters)
    clusters.fit(df_scaler.iloc[:,1:18])
    cluster_errors.append(clusters.inertia_)


# In[35]:


clusters_df=pd.DataFrame({'num_clusters':cluster_range,'cluster_errors':cluster_errors})


# In[36]:


plt.plot(cluster_range,cluster_errors,marker='o')


# In[37]:


clusters=KMeans(3)
clusters.fit(df_cleaned.iloc[:,1:18])


# In[38]:


df_cleaned['cluster']=clusters.labels_
df_cleaned['cluster'].value_counts()


# In[39]:


df_cluster_0=df_cleaned[df_cleaned['cluster']==0]
df_cluster_1=df_cleaned[df_cleaned['cluster']==1]
df_cluster_2=df_cleaned[df_cleaned['cluster']==2]


# In[40]:


df_cluster_0_ability_utilization=df_cluster_0.groupby(['ability','Utilization_category'])['Utilization_category'].count()
df_cluster_1_ability_utilization=df_cluster_1.groupby(['ability','Utilization_category'])['Utilization_category'].count()
df_cluster_2_ability_utilization=df_cluster_2.groupby(['ability','Utilization_category'])['Utilization_category'].count()


# In[41]:


df_cluster_0_ability_utilization


# In[42]:


df_cluster_1_ability_utilization


# In[43]:


df_cluster_2_ability_utilization


# In[44]:


h_clusters=AgglomerativeClustering(3)
h_clusters.fit(df_scaler.iloc[:,1:18])
df_cleaned['h_cluster']=h_clusters.labels_


# In[45]:


df_cleaned['h_cluster'].value_counts()


# In[46]:


df_h_cluster_0=df_cleaned[df_cleaned['h_cluster']==0]
df_h_cluster_1=df_cleaned[df_cleaned['h_cluster']==1]
df_h_cluster_2=df_cleaned[df_cleaned['h_cluster']==2]


# In[47]:


df_h_cluster_0_ability_utilization=df_h_cluster_0.groupby(['ability','Utilization_category'])['Utilization_category'].count()
df_h_cluster_1_ability_utilization=df_h_cluster_1.groupby(['ability','Utilization_category'])['Utilization_category'].count()
df_h_cluster_2_ability_utilization=df_h_cluster_2.groupby(['ability','Utilization_category'])['Utilization_category'].count()


# In[48]:


df_h_cluster_0_ability_utilization


# In[49]:


df_h_cluster_1_ability_utilization


# In[50]:


df_h_cluster_2_ability_utilization


# In[51]:


df_cleaned['test']=np.where(df_cleaned['cluster']==df_cleaned['h_cluster'],'true','false')
df_cleaned['test'].value_counts()


# In[52]:


df_cleaned_target['Z_score']=zscore(df_cleaned_target.PURCHASES_FREQUENCY)
df_cleaned_target[(df_cleaned_target.Z_score<-3)|(df_cleaned_target.Z_score>3)]


# In[53]:


x_train,x_test,y_train,y_test=train_test_split(sm.add_constant(df_cleaned_target[['ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY']]),df_cleaned_target['PURCHASES_FREQUENCY'],test_size=0.2,random_state=42)


# In[54]:


model_1=sm.OLS(y_train,x_train).fit()
model_1.summary()


# In[55]:


sns.heatmap(df_cleaned_target[['ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY']].corr(),annot=True)
plt.autoscale(enable=True,axis='y')


# In[56]:


x_train,x_test,y_train,y_test=train_test_split(sm.add_constant(df_cleaned_target[['ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY']]),df_cleaned_target['PURCHASES_FREQUENCY'],test_size=0.2,random_state=42)


# In[57]:


model_2=sm.OLS(y_train,x_train).fit()
model_2.summary()


# In[58]:


probplot=sm.ProbPlot(model_2.resid)
probplot.ppplot(line='45')
plt.show()


# In[59]:


def get_standardized_values(x):
    return (x-x.mean())/x.std()
plt.scatter(get_standardized_values(model_2.fittedvalues),get_standardized_values(model_2.resid))
plt.show()


# In[60]:


df_cleaned_target['cube_target']=np.power(df_cleaned_target['PURCHASES_FREQUENCY'],1/3)


# In[87]:


x_train,x_test,y_train,y_test=train_test_split(sm.add_constant(df_cleaned_target[['ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY']]),df_cleaned_target['cube_target'],test_size=0.2,random_state=42)


# In[88]:


model_3=sm.OLS(y_train_cube,x_train_cube).fit()
model_3.summary()


# In[63]:


plt.scatter(get_standardized_values(model_3.fittedvalues),get_standardized_values(model_3.resid))
plt.show()


# In[89]:


x_train,x_test,y_train,y_test=train_test_split(sm.add_constant(df_cleaned_target[['ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY']]),df_cleaned_target['PURCHASES_FREQUENCY'],test_size=0.2,random_state=42)


# In[90]:


df_cleaned_target['model-2_resid']=model_2.resid
np.round(df_cleaned_target['model-2_resid'].describe(),2)


# In[91]:


def get_train_test_rmse(model):
    y_train_pred=model.predict(x_train)
    rmse_train=metrics.mean_squared_error(y_train,y_train_pred)
    y_test_pred=model.predict(x_test)
    rmse_test=metrics.mean_squared_error(y_test,y_test_pred)
    print('train_score',rmse_train,'test_score',rmse_test)


# In[92]:


get_train_test_rmse(model_2)


# In[93]:


r2_ols=metrics.r2_score(y_test,model_2.predict(x_test))
print('R squared value',r2_ols)


# In[94]:


params={'alpha':range(1,10)}
ridge=Ridge()
clf=GridSearchCV(ridge,params,cv=5,scoring = 'neg_mean_squared_error')
clf.fit(x_train,y_train)
clf.best_params_


# In[95]:


ridge=Ridge(alpha=1,max_iter=500)
ridge.fit(x_train,y_train)
get_train_test_rmse(ridge)
ridge.coef_


# In[96]:


r2=metrics.r2_score(y_train,ridge.predict(x_train))
print('R squared value',r2)


# In[97]:


r2=metrics.r2_score(y_test,ridge.predict(x_test))
print('R squared value',r2)


# In[98]:


params={'alpha':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.01,range(1,10)]}
lasso=Lasso()
clf_lasso=GridSearchCV(lasso,params,cv=5,scoring = 'neg_mean_squared_error')
clf_lasso.fit(x_train,y_train)
clf_lasso.best_params_


# In[99]:


lasso=Lasso(alpha=0.01,max_iter=500)
lasso.fit(x_train,y_train)
get_train_test_rmse(lasso)
lasso.coef_


# In[100]:


r2_lasso=metrics.r2_score(y_train,clf_lasso.predict(x_train))
print('R squared value',r2_lasso)


# In[101]:


r2_lasso_test=metrics.r2_score(y_test,clf_lasso.predict(x_test))
print('R squared value',r2_lasso_test)


# In[102]:


enet=ElasticNet(alpha=1.01,l1_ratio=0.001,max_iter=500)
enet.fit(x_train,y_train)
get_train_test_rmse(enet)
enet.coef_


# In[103]:


r2_enet=metrics.r2_score(y_train,enet.predict(x_train))
print('R squared value',r2_enet)


# In[104]:


r2_enet_test=metrics.r2_score(y_test,enet.predict(x_test))
print('R squared value',r2_enet_test)


# In[105]:


r2=metrics.r2_score(y_test,ridge.predict(x_test))
print('R squared value',r2)


# In[106]:


scores_cross_fold_r=cross_val_score(ridge,x_train,y_train,cv=10,scoring='r2')
print('mean R squared values',np.mean(scores_cross_fold_r),'Std Deviation of prediction',np.std(scores_cross_fold_r))


# In[107]:


scores_cross_fold=cross_val_score(ridge,x_train,y_train,cv=10,scoring='neg_mean_squared_error')
print('mean error',np.mean(scores_cross_fold),'Std Deviation of error',np.std(scores_cross_fold))


# In[108]:


linreg=LinearRegression()
linreg.fit(x_train,y_train)


# In[109]:


scores_linreg_r=cross_val_score(linreg,x_train,y_train,cv=10,scoring='r2')
print('mean R squared values',np.mean(scores_linreg_r),'Std Deviation of prediction',np.std(scores_linreg_r))


# In[110]:


scores_linreg=cross_val_score(linreg,x_train,y_train,cv=10,scoring='neg_mean_squared_error')
print('mean error',np.mean(scores_linreg),'Std Deviation of error',np.std(scores_linreg))


# In[ ]:





# In[ ]:




