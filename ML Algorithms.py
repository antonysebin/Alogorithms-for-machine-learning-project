#!/usr/bin/env python
# coding: utf-8

# In[7]:


#simple linear regression
import numpy as np
import pandas as pd 


# In[32]:


df=pd.read_csv("WeatherData.csv")
df.head(5)


# In[33]:


x= df.iloc[:, :-1].values # hours, independent variable
y= df.iloc[:, 1].values # score, dependent variable
x1=pd.DataFrame(x)
print("Temperature")
print(x1)
y1=pd.DataFrame(y)
print("\n","Humidity")
print(y1)


# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)
y_pred= regressor.predict(x_test)
x_pred= regressor.predict(x_train)
df2 = pd.DataFrame({'Actual Y-Data': y_test, 'Predicted Y-Data':
y_pred})
print(df2)


# In[24]:


#multiple linear regression
dr=pd.read_csv("WeatherDatam.csv")
dr.head(5)


# In[25]:


x= dr.iloc[:, :-1].values
y= dr.iloc[:, 3].values
df2=pd.DataFrame(x)
print("X=")
print(df2.to_string())
df3=pd.DataFrame(y)
print("Y=")
print(df3.to_string())


# In[29]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x= LabelEncoder()
x[:, 2]= labelencoder_x.fit_transform(x[:,2])
# State column
ct = ColumnTransformer([("State", OneHotEncoder(), [2])], remainder =
'passthrough')

x = x[:, 1:]
df4=pd.DataFrame(x)
print("Updated X=")
print(df4.to_string())


# In[31]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)
#Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)
#Predicting the Test set result;
y_pred= regressor.predict(x_test)
#To compare the actual output values for X_test with the predicted value
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.to_string())
print("Mean")
print(dr.describe())
print("-------------------------------------")


# In[36]:


import numpy as np
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[11]:


#polynomial regerssion

df=pd.read_csv("WeatherDatap.csv")
df.head(5)


# In[15]:


x= df.iloc[:, 0:1].values
y= df.iloc[:, 1].values
df1=pd.DataFrame(x)
df2=pd.DataFrame(y)
print("pressure")
print(df1.to_string())
print("humidity")
print(df2.to_string())


# In[17]:


from sklearn.linear_model import LinearRegression
lin_regs= LinearRegression()
lin_regs.fit(x,y)


# In[19]:


from sklearn.preprocessing import PolynomialFeatures
poly_regs= PolynomialFeatures(degree= 2)
x_poly= poly_regs.fit_transform(x)
lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly, y)


# In[25]:


import matplotlib.pyplot as mtp 
mtp.scatter(x,y,color="blue")
mtp.plot(x,lin_regs.predict(x), color="red")
mtp.title("Bluff detection model(Linear Regression)")
mtp.xlabel("pressure")
mtp.ylabel("humidity")
mtp.show()


# In[26]:


lin_pred = lin_regs.predict([[6.5]])
print(lin_pred)
poly_pred = lin_reg_2.predict(poly_regs.fit_transform([[6.5]]))
print(poly_pred)


# In[27]:


#Logistic Regression
import pandas as pd
df=pd.read_csv("weatherAlbury.csv")
df.head(5)


# In[28]:


df.isnull().sum()
df.fillna(df.mean(), inplace=True)
df.head(5)


# In[29]:


df.drop(['Date', 'Location'], axis=1, inplace=True)
df.head(5)


# In[30]:


df.RainToday = [1 if each == 'Yes' else 0 for each in df.RainToday]
df.RainTomorrow = [1 if each == 'Yes' else 0 for each in df.RainTomorrow]
df.sample(3)


# In[36]:


Y = df.RainTomorrow.values
X = df.drop('RainTomorrow', axis=1)
X.head()


# In[37]:


import numpy as np
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
df2=pd.DataFrame(X_test)
#test data
print(df2.to_string())
#pred. data
print(y_pred)
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df2.to_string())
#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,
y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[38]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,
y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[5]:


#K-means Clustering
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.metrics import accuracy_score
df=pd.read_csv("IrisData.csv")
df.head(5)


# In[8]:


x= df.iloc[:, [2,4]].values 
y= df.iloc[:, 5].values
df2=pd.DataFrame(x)
print(df2.to_string())


# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
print("x_train b4 scaling..")
df3=pd.DataFrame(x_train)
print(df3.to_string())


# In[10]:


from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
print("x_train after scaling...")
df4=pd.DataFrame(x_train)
print(df4.to_string())


# In[11]:


from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
classifier.fit(x_train, y_train)
y_pred= classifier.predict(x_test)
print(y_pred)
print("Prediction comparison")
ddf=pd.DataFrame({"Y_test":y_test,"Y-pred":y_pred})
print(ddf.to_string())
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))


# In[53]:


#Support Vector Machine
df=pd.read_csv("apples_and_oranges.csv")
df.head(5)


# In[71]:


x= df.iloc[:, [0,1]]
y= df.iloc[:, [2]]


# In[90]:


x


# In[91]:


y


# In[94]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, encoded, test_size= 0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
print(y_train)
#print(x_train)

from sklearn.svm import SVC 
classifier.fit(x_train, y_train)
y_pred= classifier.predict(x_test)

df2=pd.DataFrame({"Actual Y_Test":y_test,"PredictionData":y_pred})
print("prediction status")
print(df2.to_string())


# In[92]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
encoded = le.fit_transform(y['Class'])


# In[95]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))


# In[43]:


#Decision Trees in Python
import pandas as pd
df=pd.read_csv("heart_disease_dataset.csv")
df.head(5)


# In[44]:



df.isnull().sum()


# In[ ]:





# In[45]:


ef=df.drop(['Sex','ExerciseAngina','Oldpeak','ST_Slope'],axis=1)
ef


# In[46]:


##Random Forest Algorithm

import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt

ds=pd.read_csv("rice_dataset.csv")
df=pd.DataFrame(ds)
df


# In[47]:


x=ds.iloc[:,-1].values
y=ds.iloc[:,11].values

df1=pd.DataFrame(x)
print(df1)
df2=pd.DataFrame(y)
print(df2)


# In[50]:


x=ds.iloc[:,[2,-1]].values
y=ds.iloc[:,11].values

print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()

x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion="entropy")

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
df1=pd.DataFrame({"Actual Y_Test":y_test,"Prediction Data":y_pred})
print("Prediction Result")
print(df1)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:%.2f"%(accuracy*100))


# In[ ]:




