#!/usr/bin/env python
# coding: utf-8

# # Import dataset and libraries

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("C:/Users/hp/Music/dataset.csv")
df


# # Exploratory Data Analysis
# 

# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.info()


# In[6]:


df['Architecture'].value_counts()


# In[7]:


df['Application Type'].value_counts()


# # Data preprocessing

# In[8]:


df.isnull().sum()


# In[9]:


df=df.apply(lambda x:x.fillna(x.value_counts().index[0]))
df


# In[10]:


# displaying the contents of the CSV file
df['Duration'].fillna(0, inplace=True)
df


# In[11]:


data=df.drop(['Project Elapsed Time','Project Inactive Time'],axis=1)
data


# In[12]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
data['Architecture']= le.fit_transform(data['Architecture']) 
data['Application Type']= le.fit_transform(data['Application Type'])
data['Development Type']= le.fit_transform(data['Development Type'])
data['Development Platform']= le.fit_transform(data['Development Platform'])
data['Language Type']= le.fit_transform(data['Language Type']) 
data['Relative Size']= le.fit_transform(data['Relative Size']) 
data['Used Methodology']= le.fit_transform(data['Used Methodology']) 
data['Agile Method Used']= le.fit_transform(data['Agile Method Used']) 
data['Resource Level']= le.fit_transform(data['Resource Level']) 
data['Package Customisation']= le.fit_transform(data['Package Customisation']) 
data['Industry Sector']= le.fit_transform(data['Industry Sector'])
data


# In[13]:


# Finding out the correlation coeffecient to find out which predictors are significant.
data.corr()


# In[14]:


binary_df = (data > 0).astype(int)
binary_df


# In[15]:


#mean calculation for effort
x=df['Effort'].mean()
print("Effort mean is : ",x)


# In[16]:


# standard deviation calculation for effort
y=data['Effort'].std()
print("Effort standardDeviation is : ",y)


# In[17]:


#normal distribution calculation for effort
import scipy.stats
z=scipy.stats.norm(5005.31, 16773.12).pdf(98)
print("Effort normal distribution is : ",z)


# In[18]:


#plotting effort data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=[5,8])
data.hist(column='Effort',bins=8)
plt.xlabel('Effort')
plt.ylabel('Frequency')
plt.title('Effort Distribution')
plt.show()


# In[19]:


# mean calculation for duration
w=data['Duration'].mean()
print("Duration mean is : ",w)


# In[20]:


# standard deviation calculation for duration
m=data['Duration'].std()
print("Duration standard deviation is : ",m)


# In[21]:


#normal distribution calculation for duration
import scipy.stats
n=scipy.stats.norm(7.04, 6.97).pdf(100)
print("Duration normal distribution is : ",n)


# In[22]:


#plotting Duration data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.figure(figsize=[10,8])
data.hist(column='Duration',bins=8)
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.title('Duration')
plt.show()


# In[23]:


#heat map representation
# import seaborn as sns
fig, ax = plt.subplots(figsize=(15,7))
sns.heatmap(binary_df.corr(), annot=True)


# #   Splitting into training and test data

# In[24]:



X = binary_df.iloc[:, 1:12].values
# y = binary_df.iloc[:, 12:14].values
y = binary_df.iloc[:, 12:14].values  # Selecting only one target column

# Now reshape y to have the shape (num_samples, 1)
# y = y.reshape(-1, 1)

# Verify shapes
print("X shape:", X.shape)
print("y shape:", y.shape)
unique_classes = np.unique(y)
print("Unique classes:", unique_classes)


# In[25]:


from sklearn.model_selection import train_test_split
# y=y.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =
0.20,random_state=45)
import numpy as np

# Assuming y_train and y_test are your original arrays
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# In[26]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Model Building
# 

# # SVM
# 

# In[27]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
# X_train=np.ravel(X_train)
svclassifier.fit(X_train, y_train)


y_pred = svclassifier.predict(X_test)


# In[28]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
print("Accuracy of svm : ",accuracy_score(y_test, y_pred))
print("confusion matrix \n",
 confusion_matrix (y_test, y_pred))


# In[29]:


print("classification report \n")
print(classification_report(y_test, y_pred))
classification_report 


# In[30]:


from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test, y_pred)
print("MAE is %.2f " %mae)


# In[31]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_pred)
print("MSE is %.2f"%mse)


# In[32]:


from sklearn.metrics import max_error
me=max_error(y_test,y_pred) 
print("ME is %.2f"%me)


# In[33]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE is %.2f" %rmse)


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =
0.20,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Cross Validation
# 

# In[35]:


X = binary_df.iloc[:, 1:12].values
y = binary_df.iloc[:, 12:14].values
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def acc(y_true, y_pred): return round(accuracy_score(y_true, y_pred),3)


# In[36]:


#cross validation purpose
scoring = {'accuracy': make_scorer(accuracy_score),'prec': 'precision'}
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
 'fp': make_scorer(fp), 'fn': make_scorer(fn),
 'accuracy': make_scorer(acc)}


# In[37]:


def display_result(result):
 print("TP: ",result['test_tp'])
 print("TN: ",result['test_tn'])
 print("FN: ",result['test_fn'])
 print("FP: ",result['test_fp'])
 print("Accuracy : ",result['test_accuracy'])


# In[38]:


clf = SVC(kernel='linear', C=1)
result=cross_validate(clf,X_train,y_train,scoring=scoring,cv=3)
display_result(result)


# # Logistic Regression

# In[39]:


#logistic regression implementation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
clf=LogisticRegression()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy is : ",accuracy_score(y_test,y_pred))
print("confusion matrix \n")
print(confusion_matrix(y_test,y_pred))


# In[40]:


#after cross validation
result=cross_validate(clf,X_train,y_train,scoring=scoring,cv=10)
display_result(result)


# # Random Forest

# In[41]:


# Import the necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Create a Random Forest Classifier with desired parameters
# You can adjust the number of estimators, max depth, and other hyperparameters as needed
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the Random Forest model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[42]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Assuming you have your feature matrix X and target variable y ready

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Perform cross-validation using KFold (5-fold by default)
# You can specify a different number of folds using the 'cv' parameter
scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')

# Print the cross-validation scores
print("Cross-validation scores:", scores)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())

from sklearn.metrics import accuracy_score, classification_report
# Generate a classification report
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)


# # Decision Tree

# In[43]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Create a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)

# Perform cross-validation using KFold (5-fold by default)
# You can specify a different number of folds using the 'cv' parameter
scores = cross_val_score(dt_classifier, X_train, y_train, cv=5, scoring='accuracy')

# Print the cross-validation scores
print("Cross-validation scores:", scores)

# Calculate and print the mean and standard deviation of the scores
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())

# Train the Decision Tree model on the full training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dt_classifier.predict(X_test)

# Evaluate the model's performance on the test data
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test data:", accuracy)
from sklearn.metrics import classification_report


# In[44]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report

# Assuming you have your feature matrix X and target variable y ready

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)

# Perform cross-validation using KFold (5-fold by default)
# You can specify a different number of folds using the 'cv' parameter
y_pred_cv = cross_val_predict(dt_classifier, X_train, y_train, cv=5)

# Train the Decision Tree model on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred_test = dt_classifier.predict(X_test)

# Generate a classification report for cross-validation
cv_report = classification_report(y_train, y_pred_cv)

# Generate a classification report for the test data
test_report = classification_report(y_test, y_pred_test)

# Print the classification reports
# print("Classification Report (Cross-Validation):\n", cv_report)
print("\nClassification Report (Test Data):\n", test_report)


# # Calculating Effort and Duration

# In[45]:


import pandas as pd
df=pd.read_csv("C:/Users/hp/Music/LOC.csv")
df


# In[46]:


df['LOC'] = pd.to_numeric(df['LOC'], errors='coerce')
df


# In[47]:


df.describe()


# In[48]:


#calculate mean for LOC
x=df['LOC'].mean()
print(int(x))


# In[49]:


# effort calculation in person months 
a=3.2
b=1.05
KLOC=469
effort=int(a*(KLOC)**b)
print("Effort","=",effort ,"(Person Months)")


# In[50]:


#duration calculation in months
c=2.5
d=0.38
Effort=2041
Duration = int(c*(Effort)**d)
print("Duration","=",Duration ,"(Months)")


# In[51]:


#persons required to complete project
effort=2041
duration=45
PersonsRequired=effort//duration
print("Persons Required", "=", PersonsRequired)


# In[52]:


# effort calculation in person months 
a=3
b=1.12
KLOC=469
effort=int(a*(KLOC)**b)
print("Effort","=",effort ,"(Person Months)")


# In[53]:


#duration calculation in months
c=2.5
d=0.35
Effort=2943
Duration = int(c*(Effort)**d)
print("Duration","=",Duration ,"(Months)")


# In[54]:


#persons required to complete project
effort=2943
duration=40
PersonsRequired=effort//duration
print("Persons Required", "=", PersonsRequired)


# In[55]:


# effort calculation in person months
a=2.8
b=1.20
KLOC=469
effort=int(a*(KLOC)**b)
print("Effort","=",effort ,"(Person Months)")


# In[56]:


#duration calculation in months
c=2.5
d=0.32
Effort=4493
Duration = int(c*(Effort)**d)
print("Duration","=",Duration ,"(Months)")


# In[57]:


#persons required to complete project
effort=4493
duration=36
PersonsRequired=effort//duration
print("Persons Required", "=", PersonsRequired)

