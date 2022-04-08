#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
tr = pd.read_csv("C:/Users/akaks/Downloads/loan_training_set.csv")


# In[2]:


tr


# In[3]:


tr.info()


# # Drop Statistically Unimportant columns

# In[4]:


tr.drop(labels=['Loan_ID'],axis=1,inplace=True)


# In[5]:


tr.isna().sum()


# # Missing data Treatment

# In[6]:


tr.Gender.value_counts()


# In[7]:


tr.Gender = tr.Gender.fillna("Male")


# In[8]:


tr.Married.value_counts()


# In[9]:


tr.Married = tr.Married.fillna("Yes")


# In[10]:


tr.Dependents = tr.Dependents.str.replace("+","")


# In[11]:


tr.Dependents.value_counts()


# In[12]:


tr.Dependents = tr.Dependents.fillna("0")


# In[13]:


tr.Education.value_counts()


# In[14]:


tr.Education = tr.Education.fillna("Graduate")


# In[15]:


tr.Self_Employed.value_counts()


# In[16]:


tr.Self_Employed = tr.Self_Employed.fillna("No")


# In[17]:


x = round(tr.ApplicantIncome.mean(),0)


# In[18]:


tr.ApplicantIncome = tr.ApplicantIncome.fillna(x)


# In[19]:


x = round(tr.CoapplicantIncome.mean(),0)


# In[20]:


tr.CoapplicantIncome = tr.CoapplicantIncome.fillna(x)


# In[21]:


x = round(tr.LoanAmount.mean(),0)


# In[22]:


tr.LoanAmount = tr.LoanAmount.fillna(x)


# In[23]:


x = tr.Loan_Amount_Term.mode()[0]


# In[24]:


tr.Loan_Amount_Term = tr.Loan_Amount_Term.fillna(x)


# In[25]:


tr.Credit_History.value_counts()


# In[26]:


tr.Credit_History = tr.Credit_History.fillna(1)


# In[27]:


tr.isna().sum()


# In[28]:


Y = tr.Loan_Status


# In[29]:


X = tr.drop("Loan_Status",axis=1)


# In[30]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Ynew= le.fit_transform(Y)


# # Exploratory Data Analysis

# In[31]:


cat = []
con = []
for i in X.columns:
    if X[i].dtype == "object":
        cat.append(i)
    else:
        con.append(i)


# In[32]:


imp_cols = []


# In[33]:


def ANOVA(df,cat,con):
    from pandas import DataFrame
    from statsmodels.api import OLS
    from statsmodels.formula.api import ols
    rel = con + " ~ " + cat
    model = ols(rel,df).fit()
    from statsmodels.stats.anova import anova_lm
    anova_results = anova_lm(model)
    Q = DataFrame(anova_results)
    a = Q['PR(>F)'][cat]
    return round(a,4)


# In[34]:


for i in con:
    q = ANOVA(tr,"Loan_Status",i)
    print("-------------")
    print("Loan_Status vs",i)
    print("Pval: ",q)
    if(q < 0.15):
        imp_cols.append(i)


# In[35]:


imp_cols


# In[36]:


from scipy.stats import chi2_contingency
def chisquare(df,cat1,cat2):
    import pandas as pd
    ct = pd.crosstab(df[cat1],df[cat2])
    a,b,c,d = chi2_contingency(ct)
    return b


# In[37]:


for i in cat:
    if(X[i].dtypes=="object"):
        x = chisquare(tr,"Loan_Status",i)
        if(x < 0.05):
            print("Loan_Status vs ",i,"-->",x)
            imp_cols.append(i)


# In[38]:


imp_cols


# # Preprocessing

# In[39]:


X.skew()


# In[40]:


from numpy import log

def skew_rem(df,col):
    q = []
    for i in df[col]:
        if(i != 0):
            q.append(log(i))
        else:
            q.append(i)
    df[col] = q


# In[41]:


skew_rem(X,'CoapplicantIncome')


# In[42]:


skew_rem(X,'ApplicantIncome')


# In[43]:


X.skew()


# In[44]:


from ML_Fun import data_prep
Xnew = data_prep(X[imp_cols])


# In[45]:


Xnew


# # Dividing Data into training and testing set

# In[46]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(Xnew,Y,test_size=0.8,random_state=21)


# # Create a ML Model

# In[47]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(xtrain,ytrain)


# # Create Predictions

# In[48]:


tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)


# In[49]:


from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc = accuracy_score(ytest,ts_pred)


# In[50]:


tr_acc


# In[51]:


ts_acc


# # Try Tree Model

# In[52]:


X_ = data_prep(X)


# In[53]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X_,Y,test_size=0.2,random_state=21)


# In[54]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
model = dtc.fit(xtrain,ytrain)

tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)

tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc = accuracy_score(ytest,ts_pred)
print(tr_acc)
print(ts_acc)


# In[55]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
def tree(dtc):
    model = dtc.fit(xtrain,ytrain)
    tr_pred = model.predict(xtrain)
    ts_pred = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = accuracy_score(ytrain,tr_pred)
    ts_acc = accuracy_score(ytest,ts_pred)
    return tr_acc , ts_acc


# In[56]:


tree(dtc)


# In[57]:


for i in range(2,30,1):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(random_state=21,max_depth=i)
    print(tree(dtc))


# In[58]:


for i in range(2,30,1):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(random_state=21,min_samples_leaf=i)
    print(tree(dtc))


# In[59]:


for i in range(2,40,1):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(random_state=21,min_samples_split=i)
    print(tree(dtc))


# # Try Adaboost

# In[60]:


from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(DecisionTreeClassifier(random_state=21,max_depth=2),n_estimators=30)
tree(adb)


# In[61]:


for i in range(2,30):
    adb = AdaBoostClassifier(DecisionTreeClassifier(random_state=21,max_depth=2),n_estimators=i)
    print(tree(adb))
    


# In[62]:


dtc = DecisionTreeClassifier(random_state=21,max_depth=2)
model = dtc.fit(X_,Y)


# # Make Predictions using Best Model

# In[63]:


ts = pd.read_csv("C:/Users/akaks/Downloads/testing_set.csv")


# In[64]:


ts.isna().sum()


# In[65]:


ts


# In[66]:


from ML_Fun import replacer
replacer(ts)


# In[67]:


ts


# In[68]:


X = ts.drop("Loan_ID",axis=1)


# In[69]:


from ML_Fun import data_prep
Xnew = data_prep(X)


# In[70]:


Xnew


# In[71]:


pred = model.predict(Xnew)


# In[72]:


ts['Predicted_Loan_Status'] = pred


# In[73]:


ts


# In[74]:


pd.DataFrame([Xnew.columns,dtc.feature_importances_]).T


# # Identify Customer Segments

# In[75]:


X = tr[['CoapplicantIncome','LoanAmount','Credit_History']]


# In[76]:


X


# In[77]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xnew = pd.DataFrame(ss.fit_transform(X),columns=X.columns)


# In[78]:


from sklearn.cluster import KMeans
KM = KMeans(n_clusters=4)
model = KM.fit(Xnew)


# In[79]:


X['Cluster'] = model.labels_


# In[80]:


X


# In[81]:


import matplotlib.pyplot as plt
plt.scatter(X.LoanAmount,X.CoapplicantIncome,c=model.labels_)


# In[82]:


plt.scatter(X.LoanAmount,X.CoapplicantIncome,c=X.Credit_History)


# In[83]:


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
w = le.fit_transform(tr.Loan_Status)


# In[84]:


plt.scatter(X.LoanAmount,X.CoapplicantIncome,c=w)


# # If customer is not eligible for the input required amount and duration:

# #### 1) what can be amount for the given duration?

# In[179]:


P1 = tr[tr.Loan_Status=='Y']


# In[180]:


P2 = ts[ts.Predicted_Loan_Status=='Y']


# In[181]:


P2 = P2.rename({"Predicted_Loan_Status":"Loan_Status"},axis=1)


# In[182]:


P2.drop("Loan_ID",axis=1,inplace=True)


# In[183]:


trd = pd.concat([P1,P2])


# In[208]:


trd.Dependents = trd.Dependents.str.replace('+',"")


# In[210]:


from ML_Fun import replacer
replacer(trd)


# In[211]:


cat = []
con = []
for i in trd.columns:
    if trd[i].dtype =='object':
        cat.append(i)
    else:
        con.append(i)


# In[212]:


cat


# In[213]:


con


# In[214]:


cat.remove('Loan_Status')
con.remove("LoanAmount")


# In[215]:


Y = trd[['LoanAmount']]


# In[216]:


X = trd.drop(labels=['LoanAmount','Loan_Status'],axis=1)


# In[217]:


X.head()


# In[218]:


from sklearn.preprocessing import StandardScaler
ss1 = StandardScaler()
X1 = pd.DataFrame(ss1.fit_transform(X[con]),columns=con)
X2 = pd.get_dummies(X[cat])
X2.index = range(0,718)


# In[219]:


Xnew = X1.join(X2)


# In[220]:


Xnew


# In[221]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(Xnew,Y,test_size=0.8,random_state=21)


# In[222]:


trd.corr()[['LoanAmount']]


# In[223]:


for i in X.columns:
    if(X[i].dtypes == "object"):
        print("------------------------")
        print("Loan AMt vs",i)
        print(ANOVA(trd,i,"LoanAmount"))


# In[224]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(xtrain,ytrain)

pred_tr = model.predict(xtrain)
pred_ts = model.predict(xtest)

from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,pred_tr)
ts_err = mean_absolute_error(ytest,pred_ts)

print(tr_err)
print(ts_err)


# # Preparing Data for Prediction

# In[225]:


test = ts[ts.Predicted_Loan_Status=="N"]


# In[226]:


test


# In[227]:


from ML_Fun import replacer
replacer(test)


# In[228]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X1 = pd.DataFrame(ss.fit_transform(test[con]),columns=con)
X2 = pd.get_dummies(test[cat])
X2.index = range(0,71)


# In[229]:


Xnew = X1.join(X2)


# In[230]:


Xnew


# In[233]:


Predicted_Loan_Amount = model.predict(Xnew)


# In[237]:


ts[ts.Predicted_Loan_Status=='N']


# # b.)if duration is less than equal to 20 years, is customer eligible for required amount for some longer duration? What is that duration?(Regression)
# 

# In[239]:


test = ts[ts.Loan_Amount_Term <= 240]


# In[242]:


test.shape


# In[243]:


test.index = range(0,34)


# In[244]:


test


# In[245]:


Y = tr[['Loan_Amount_Term']]


# In[248]:


X = tr.drop(labels=['Loan_Amount_Term','Loan_Status'],axis=1)


# In[249]:


X


# In[250]:


Xnew = data_prep(X)


# In[251]:


Xnew


# In[252]:


xtrain,xtest,ytrain,ytest = train_test_split(Xnew,Y,test_size=0.8,random_state=21)


# In[253]:


model = lr.fit(xtrain,ytrain)


# In[256]:


pred_tr = model.predict(xtrain)
pred_ts = model.predict(xtest)

tr_err = mean_absolute_error(ytrain,pred_tr)
ts_err = mean_absolute_error(ytest,pred_ts)

print(tr_err)
print(ts_err)


# In[257]:


replacer(test)


# In[258]:


X  = test.drop(labels=['Loan_ID','Loan_Amount_Term','Predicted_Loan_Status'],axis=1)


# In[260]:


Xnew = data_prep(X)


# In[262]:


Xnew.shape


# In[263]:


model.predict(Xnew)


# In[ ]:




