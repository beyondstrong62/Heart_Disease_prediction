#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[35]:


import pandas as pd                               #for data manipulation
import matplotlib.pyplot as plt                   #for data visualization
import seaborn as sns                             #for data visualization
import numpy as np


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# In[36]:


data=pd.read_csv("heart.csv")


# In[37]:


print(data.shape)


# In[38]:


data.head()


# In[39]:


data.tail()


# In[40]:


pd.set_option("display.float", "{:.2f}".format)
data.describe()


# In[41]:


data.isna().sum()


# In[42]:


count_Male = len(data[data.sex == 1])
count_Female = len(data[data.sex == 0])
print("Total Number of Male Patients = ",count_Male)
print("\nPercentage of Male Patients = {:.2f}%".format((count_Male)/(len(data.sex))*100))
print("\nTotal Number of Female Patients = ",count_Female)
print("\nPercentage of Female Patients = {:.2f}%".format((count_Female)/(len(data.sex))*100))


# In[43]:


plt.figure(figsize=(10,8))
sns.countplot(x='sex', data=data, palette='Set1')
plt.xticks(ticks=[1, 0], labels = ["Male", "Female"])
plt.title("No. of Males and Females present in the dataset", size=15)
plt.show()


# In[44]:


m=0
f=0
count_disease = len(data[data.target == 1])
count_nodisease = len(data[data.target == 0])
print("Total Number of Patients having Heart Diseases =  ",count_disease)
for i in data.index:
    if(data['sex'][i]==1 and data['target'][i]==1):
        m=m+1
    else:
        continue
print("\nNo. of Male patient suffering from heart disease = ",m)
for i in data.index:
    if(data['sex'][i]==0 and data['target'][i]==1):
        f=f+1
    else:
        continue
print("\nNo. of female patient suffering from heart disease = ",f)
print("\nPercentage of Patients Have Heart Disease = {:.2f}%".format((count_disease / (len(data.target))*100)))
print("\nTotal Number of Patients have not Heart Diseases = ",count_nodisease)
print("\nPercentage of Patients Haven't Heart Disease = {:.2f}%".format((count_nodisease / (len(data.target))*100)))


# In[45]:


plt.figure(figsize=(10,8))
sns.countplot(x='target', data=data, palette='Set1')
plt.xticks(ticks=[1, 0], labels = ["Have Heart Disease", "No Heart Disease"])
plt.title("No. of Patient in the dataset", size=15)
plt.show()


# In[46]:


labels = ['Yes', 'No']
values = data['target'].value_counts().values
plt.rcParams["figure.figsize"] = (8,4)
plt.pie(values, labels=labels, autopct='%1.1f%%', textprops = {"fontsize" : 15})
plt.title('Heart Disease', fontsize=20)
plt.show()
data_gender = data.groupby(["sex","target"]).size() 
plt.pie(data_gender.values, labels = ["Female, No_Heart_disease", "Female, With_heart_disease", 
                                    "Male, No_Heart_disease", "Male, With_Heart_disease"],autopct='%1.1f%%',radius = 1.5, 
        textprops = {"fontsize" : 15})
plt.show()


# In[47]:


categorical_val = []
continous_val = []
for column in data.columns:
    if len(data[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


# In[48]:



print("Categorical Values are = ",categorical_val)
print("Continous Values are = ",continous_val)


# In[49]:



plt.figure(figsize=(15, 15))

for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    data[data["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    data[data["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[50]:



plt.figure(figsize=(15, 15))

for i, column in enumerate(continous_val, 1):
    plt.subplot(3, 2, i)
    data[data["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    data[data["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[51]:



# Create another figure
plt.figure(figsize=(10, 8))

# Scatter with postivie examples
plt.scatter(data.age[data.target==1],
            data.thalach[data.target==1],
            c="red")

# Scatter with negative examples
plt.scatter(data.age[data.target==0],
            data.thalach[data.target==0],
            c="blue")

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);


# In[52]:


corr_matrix = data.corr()
fig, ax = plt.subplots(figsize=(15, 15))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[53]:


categorical_val.remove('target')
dataset = pd.get_dummies(data, columns = categorical_val)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
dataset[continous_val] = sc.fit_transform(dataset[continous_val])


# In[54]:


from sklearn.model_selection import train_test_split

#Spliting the Features And Target
X = data.drop('target', axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)   #80% training and 20% testing
print("Training features have {0} records and Testing features have {1} records.".      format(X_train.shape[0], X_test.shape[0]))


# In[55]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, ShuffleSplit
import pandas as pd

# Your find_best_model function and its call


# In[56]:


def find_best_model(X, y):
    models = {
        'logistic_regression': {
            'model': LogisticRegression(solver='lbfgs', multi_class='auto'),
            'parameters': {
                'C': [1,5,10]
               }
        },
        
        'decision_tree': {
            'model': DecisionTreeClassifier(splitter='best'),
            'parameters': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5,10]
            }
        },
        
        'random_forest': {
            'model': RandomForestClassifier(criterion='gini'),
            'parameters': {
                'n_estimators': [10,15,20,50,100,200]
            }
        },
        'svm': {
            'model': SVC(gamma='auto'),
            'parameters': {
                'C': [1,10,20],
                'kernel': ['rbf','linear']
            }
        }

    }
    
    scores = [] 
    cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
        
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = cv_shuffle, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
        
    return pd.DataFrame(scores, columns=['model','best_parameters','score'])

find_best_model(X_train, y_train)
                


# In[57]:


# Using cross_val_score for gaining average accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(RandomForestClassifier(n_estimators=20, random_state=0), X_train, y_train, cv=5)
print('Average Accuracy : {}%'.format(round(sum(scores)*100/len(scores)), 3))


# In[58]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("\n\nTrain Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("\n\nTest Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[59]:


from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver="liblinear")
lr_clf.fit(X_train, y_train)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print("Training set score: {:.3f}".format(lr_clf.score(X_train, y_train)))
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)
print("Test set score: {:.3f}".format(lr_clf.score(X_test, y_test)))


# In[60]:


test_score = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, lr_clf.predict(X_train)) * 100

results_df = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df


# In[61]:


# Taking the input from user 64,1,3,110,211,0,0,144,1,1.8,1,0,2,
print("Please enter the following detail")
input_data=(int(input("\nAge\n\n")),
            int(input("\nGender\n\n0 for 'Female' \n1 for 'Male'\n\n")),
            int(input("\nChest Pain measurefrom (0-4)\n1: typical angina\n2: atypical angina\n3: non-anginal pain\n4: asymptomatic\n\n")),
            int(input("\nResting Blood Pressure value\n\n")),
            int(input("\nserum cholestoral in mg/dl\n\n")),
            int(input("\nFasting Blood Sugar\n0 for 'lower than 120mg/ml'\n1 for 'greater than 120mg/ml'\n\n")),
            int(input("\nResting Electrocardiographic(ECG) results\n0 for 'normal'\n1 for 'ST-T wave abnormality'\n2 for 'left ventricular hypertrophy'\n\n")),
            int(input("\nMaximum heart rate achieved\n\n")),
            int(input("\nExercised induced Agina\n0 for 'no'\n1 for 'yes'\n\n")),
            float(input("\nST depression induced by exercise relative to rest (0.0-10.0)\n\n")),
            int(input("\nThe slope of the peak exercise ST segment\n1 for 'upsloping'\n2 for 'flat'\n3 for 'downsloping'\n\n")),
            int(input("\nNumber of major vessels (0-4) colored by flourosopy\n\n")),
            int(input("\nThalessemia Value\nthal: 3 = normal; 6 = fixed defect; 7 = reversable defect\n\n"))
            )


# In[63]:


input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction = lr_clf.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print("\nThis person's heart is healthy 💛💛💛💛")
else:
    print("\nThis person is suffering from heart disease 💔💔💔💔")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




