#!/usr/bin/env python
# coding: utf-8


# ## a. Content
1.0 Data Cleaning
    1.1 Training Data
    1.2 Testing Data
    1.3 Training Data Encoder
    1.4 Testing Data Encoder
2.0 Exploratory Data Analysis
    2.1 Figures
        2.1.1 Figure 1 (Heat Map of Missing Data)
        2.1.2 Figure 2 (Sample observation distribution amond 
        IncomBracket)
        2.1.3 Figure 3 (Age relation to income bracket
        2.1.4 Figure 4 (Income of people with different education)
    2.2 Visualize the order of feature importance
3.0 Feature Selction
    3.1 Using Decision Tree Classifier
        3.1.1 Creating an New Feature (HoursPerWekk) - Training Data
        3.1.2 Creating an Aditional Feature (HoursPerWeek) - Testing Data
        3.1.3 Calculating after dropping IncomeBracker
    3.2 Removing Features with low importance ratio
4.0 Model Implementation
    4.1 Logistic Regrasion
        4.1.1 Logistic Regrassion (Test Data)
        4.1.2 Logistic Regrassion (Trainning Data)
    4.2 kNN Classifier
        4.2.1 kNN Classifier (Testing Data)
        4.2.2 kNN Classifier (Training Data)
    4.3 Random Forest Classifier
        4.3.1 Random Forest Classifier (Testing Data)
        4.3.2 Random Forest CLassifier (Training Data)
    4.4 Decision Tree Classifier
        4.4.1 Decision Tree Classifier (Testing Data)
        4.4.2 Decision Tree Classifier (Training Data)
5.0 Model TUning
    5.1 Logistic Regression (Model Tuning)
    5.2 kNN Classifier (Model Tunning) 
    5.3 Random Forest Classifier (Model Tunning)
    5.4 Tree Classifier (Model Tunning)
6.0 Testing
# ## b. Importing Libraries

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder as l_encoder
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import GridSearchCV
except:
    from sklearn.grid_search import GridSearchCV

#!pip install xgboost
from xgboost import XGBClassifier

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

#!pip install seaborn
#pip3 install seaborn 


# ## 1.0 Data Cleaning

# We are replacing the missing values with the most used value in the category. This means that the replaced value is a good estimation but not necessary the right method. An alternate method which might be better is estimate based on the percentage of the repeated values. For exmaple lets take into considation 10 data with 3 values as X, Y, Z. X was repeated 5 times, Y repeated 3 times and finally Z repeated 2 times. This means we will use this method of replacing missing value with 50% with X, 30% with Y and 20% with Z. 
# 
# The data can be missing due to several reasons, such as,
# 1. Privacy
# 2. Not all of the people are occupied
# 3. Error while collecting data
# 4. Not one of the options for that person

# ### 1.1 Training Data

# In[2]:


# Opening and reading data files (Training Data)

training_data = pd.read_csv("income-training.csv")
filled_training_data = training_data.apply(lambda cur_data:cur_data.fillna(cur_data.value_counts().index[0]))
fTraining_data = filled_training_data
filled_training_data.head()


# In[3]:


training_data.info()


# ### 1.2 Testing Data

# In[4]:


# Opening and reading data files (Testing Data)
testing_data = pd.read_csv("income-testing.csv")
filled_testing_data = testing_data.apply(lambda cur_data:cur_data.fillna(cur_data.value_counts().index[0]))
filled_testing_data.head()


# In[5]:


testing_data.info()


# ### 1.3 Training Data Encoding

# In[6]:


# Training Data
# Using label encoder (imported in libraries section)
label_encoder = l_encoder()

for cur_col in filled_training_data:
    if filled_training_data[cur_col].dtypes=='object':
            data = filled_training_data[cur_col]
            label_encoder.fit(data.values)
            filled_training_data[cur_col]=label_encoder.transform(filled_training_data[cur_col])
            
filled_training_data.head()


# ### 1.4 Testing Data Encoding

# In[7]:


# Testing Data
# Using label encoder (imported in libraries section)
for cur_col in filled_testing_data:
    if filled_testing_data[cur_col].dtypes=='object':
            data = filled_testing_data[cur_col]
            label_encoder.fit(data.values)
            filled_testing_data[cur_col]=label_encoder.transform(filled_testing_data[cur_col])
            
filled_testing_data.head()


# ## 2.0 Exploratory data analysis

# ### 2.1 Figures

# ### 2.1.1 Figure 1 (Heat Map of Missing Data)

# In[8]:


# Heat Map of Missing Data
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(training_data.isnull(), cmap='binary', yticklabels=False, cbar=False, ax=ax)
plt.show()


# Figure 1, illustrate the columns with missing data. Due to the size of the data, these missing data will not have a huge impact on our analysis.

# ### 2.1.2 Figure 2 (Sample observation distribution among IncomeBracket)

# In[9]:


# Sample observations distribution among the three income bracket categories
sns.countplot(x='IncomeBracket',hue='Sex',data = training_data)
# Axes labels and title
plt.ylabel('Number of People')
plt.title('Total Number of People Vs. Income Bracket (Figure 1)')

plt.show()


# In figure 2, we can see the distribution of people among the three income bracket categories. We can indicate that number or people that receiving salary <50K is higher than the other income categories with 50-100k second and >100K third in ranking.  Furthermore, we can notice that Male has higher income compared to Female in all of the categories.  Also, we can notice the ratio of Female to Male in <50K is higher than the other two categories.

# ### 2.1.3 Figure 3 (Age relation to income bracket)

# In[10]:


# Age relation to income bracket
sns.boxplot(x='IncomeBracket', y='Age', data = training_data)
# Axes labels and title
plt.ylabel('Age')
plt.title('Distribution of Age Vs. Income Bracket (Figure 2)')

plt.show()


# In figure 3, we can notice that the people in the lowest income bracket are younger than the other two brackets. Furthermore, people in the highest income bracket are slightly older than the people in 50-100k bracket. This is a good observation of the differences between the three classes.

# ### 2.1.4 Figure 4 (Income of people with different educaiton)

# In[11]:


# Income of the people with different education
fig = plt.figure(figsize=(10,10))

incomeBracket = training_data["IncomeBracket"]
educationData = training_data["Education"]

educationIncomeDataU50 = educationData[incomeBracket=="<50K"]
educationIncomeDataU100 = educationData[incomeBracket=="50-100K"]
educationIncomeDataO100 = educationData[incomeBracket==">100K"]

educationU50Count = educationIncomeDataU50.value_counts()
educationU100Count = educationIncomeDataU100.value_counts()
educationO100Count = educationIncomeDataO100.value_counts()
educationU50Count.plot.bar(rot=0, color='blue')
educationU100Count.plot.bar(rot=0, color='orange')
educationO100Count.plot.bar(rot=0, color='green')

# Axes labels and title
plt.xticks(rotation='vertical')
plt.xlabel('Education')
plt.ylabel('Number of People')
plt.title('Income of Number of People with Different Education (Figure 3)')
# Legend labels and title
blue_patch = mpatches.Patch(color='blue', label=' Salary <50k')
orange_patch = mpatches.Patch(color='orange', label='Salary 50-100k')
green_patch = mpatches.Patch(color='green', label='Salary >100k')
plt.legend(handles=[blue_patch, orange_patch, green_patch])

plt.show()


# In figure 4, we can notice that the majority of the people are with bachelor degree and the second highest are high school degree. Furthermore, we can notice that the education level, for example bachelor compared to HS-grad has almost the same percentage of people with 50-100K income. This is an important graph to show the highest and lowest education level for determining the category.

# ### 2.2 Visualize the order of feature importance

# The most important feature is showing in the following section

# In[12]:


# Importance of the features 
incomeBracket = filled_training_data["IncomeBracket"]
droppedIncomeData = filled_training_data.drop(['IncomeBracket'],axis=1)

# load the iris datasets
dataset = datasets.load_iris()

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(droppedIncomeData, incomeBracket)

for column_name, importance in zip(droppedIncomeData.columns.values, model.feature_importances_):
    print(column_name, importance)

# Plotting the ratio of importance
width = 1/1.5
plt.bar(droppedIncomeData.columns.values, model.feature_importances_, width, color="blue")

plt.xticks(rotation='vertical')
plt.xlabel('Features')
plt.ylabel('Ratio')
plt.title('Importance of the Categories (Figure 5)')


# Based on figure 5, we can notice that the importance of FinalWeight, and Age is higer than other categories.

# ## 3.0 Feature selection

# Proper engineering could boost model performance and can reduce model's inherit weakness. Furthermore engineered non linear features effectively remove linear models. With proper feature selection, feature engineering can be without parameters which lead to remving the bias.

# ### 3.1 Using Decision Tree Classifier

# Finding the importance of each feature using Decision Tree Classifier

# ### 3.1.1 Creating an New Feature (HoursPerWeek) - Training Data

# In[13]:


# Creating a new column
# Employment job type
employmentTypes = []

cutOffHours = 35

for hoursPWeek in training_data["HoursPerWeek"]:
    # sometimes employees they remove the 5 hours of lunch
    if hoursPWeek < 35: 
        employmentTypes.append("Part Time")
    else:
        employmentTypes.append("Full Time")
        
temp_dataFrame = pd.DataFrame({"EmploymentType": employmentTypes})
training_data_n = training_data.join(temp_dataFrame)

filled_training_data_n = training_data_n.apply(lambda cur_data:cur_data.fillna(cur_data.value_counts().index[0]))

for cur_col in filled_training_data_n:
    if filled_training_data_n[cur_col].dtypes=='object':
            data = filled_training_data_n[cur_col]
            label_encoder.fit(data.values)
            filled_training_data_n[cur_col]=label_encoder.transform(filled_training_data_n[cur_col])
            
filled_training_data_n.head()


# ### 3.1.2 Creating an Aditional Feature (HoursPerWeek) - Testing Data

# In[14]:


# Creating a new column
# Employment job type
employmentTypes = []

cutOffHours = 35

for hoursPWeek in testing_data["HoursPerWeek"]:
    # sometimes employees they remove the 5 hours of lunch
    if hoursPWeek < 35:
        employmentTypes.append("Part Time")
    else:
        employmentTypes.append("Full Time")
        

temp_dataFrame = pd.DataFrame({"EmploymentType": employmentTypes})
testing_data_n = testing_data.join(temp_dataFrame)

filled_testing_data_n = testing_data_n.apply(lambda cur_data:cur_data.fillna(cur_data.value_counts().index[0]))

for cur_col in filled_testing_data_n:
    if filled_testing_data_n[cur_col].dtypes=='object':
            data = filled_testing_data_n[cur_col]
            label_encoder.fit(data.values)
            filled_testing_data_n[cur_col]=label_encoder.transform(filled_testing_data_n[cur_col])
            
filled_testing_data_n.head()


# ### 3.1.3 Calculating After Dropping IncomeBracket

# In[15]:


# Importacne of the feature 
incomeBracket = filled_training_data_n["IncomeBracket"]
droppedIncomeData = filled_training_data_n.drop(['IncomeBracket'],axis=1)

# load the iris datasets
dataset = datasets.load_iris()

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(droppedIncomeData, incomeBracket)

importanceData = []

for column_name, importance in zip(droppedIncomeData.columns.values, model.feature_importances_):
    print(column_name, importance)
    importanceData.append([column_name, importance])
    
width = 1/1.5
plt.bar(droppedIncomeData.columns.values, model.feature_importances_, width, color="blue")

plt.xticks(rotation='vertical')
plt.xlabel('Features')
plt.ylabel('Ratio')
plt.title('Importance of the Categories (Figure 6)')


# Above graph is showing the impotence of the features and what ratio they have. We can notice that some features have a low ratio, and if we remove them it is better for the analysis in next parts, because we can have a higher accuracy in each algorithm. 

# ### 3.2 Removing Features with Low Importance Ratio (Less than 0.025)

# In[16]:


newData = filled_training_data_n
newTestData = filled_testing_data_n

# Creating loop to filter using importance threshold
for column_name, importance in importanceData:
    if importance < 0.025:
        newData = newData.drop([column_name],axis=1)
        newTestData = newTestData.drop([column_name],axis=1)
        
y_data = newData["IncomeBracket"]
x_data = newData.drop(["IncomeBracket"],axis=1)
x_data.head()

y_test_data = newTestData["IncomeBracket"]
x_test_data = newTestData.drop(["IncomeBracket"],axis=1)
x_test_data.head()


# Above is our feature importance after we remove the features which has ratio less than 0.025. We can realize that now we are only taking into consideration the higher importance features in next parts.

# ## 4.0 Model Implementation

# In this section we will use the following algorithm, 
# 
# 1. Logistic Regression
# 2. kNN Classifier
# 3. Random Forrest
# 4. Decision Tree
# 
# We are using these classifiers because we can have higher accuracy. We calculate before and after K-fold method. 

# ### 4.1 Logistic Regression

# One of the important reason to choose Logistic Regression is the high accuracy. Also, by using this classifier we will calculate the probability with a good binary classification problem. Furthermore, we can do the prediction of the categorical target variable.

# ### 4.1.1 Logistic Regrassion (Test Data)

# In[17]:


# Logistic Regression
# Applying Logistic Regression for Test Data
logisticReg = LogisticRegression()
# Fitting Logistic Regrassion into the data
logisticReg.fit(x_data, y_data)

logisticRegPred = logisticReg.predict(x_test_data)
print(accuracy_score(y_test_data, logisticRegPred))


# ### 4.1.2 Logistic Regrassion (Training Data)

# In[18]:


Y = filled_training_data_n["IncomeBracket"]
X = filled_training_data_n.drop(["IncomeBracket"], axis = 1)


# In[19]:


# Applying Logistic Regression for Training Data
training_data_len = len(training_data)
# Using K-fold
kFold = KFold(training_data_len, n_folds=10)

exitingResults = []

curInd = 0

for trainId, testId in kFold:
    x_model, x_test_model = X.values[trainId], X.values[testId]
    y_model, y_test_model = Y.values[trainId], Y.values[testId]
    # Fitting Logistic Regrassion into the data
    logisticReg.fit(x_model, y_model)
    predictions = logisticReg.predict(x_test_model)
    accuracy = accuracy_score(y_test_model, predictions)
    exitingResults.append(accuracy)
    curInd += 1
    print("Fold ID : %d Accuracy : %.6f" % (curInd, accuracy))   
    
mean_outcome = np.mean(exitingResults)
print("Mean Accuracy : %.6f" %mean_outcome ) 


# ### 4.2 kNN Classifier

# One of the advantages of kNN Classifier is it does not make assumptions about the data we are evaluating. It means that we are labelling the data (our Training Data) to determine the label for the new datal. This algorithm as it is known from its name, it use K nearest training examples to determine new data point nonparametric.

# ### 4.2.1 kNN Classifier (Testing Data)

# In[20]:


# Knn Classifier
# Applying kNN Classifier for Test Data

knnClf = KNeighborsClassifier(n_neighbors=10)
# Fitting kNN Classifier to the data
knnClf.fit(x_data, y_data)

knnPred = knnClf.predict(x_test_data)
print(accuracy_score(y_test_data, knnPred))


# ### 4.2.2 kNN Classifier (Training Data)

# In[21]:


# Applying kNN Classifier for Test Data
training_data_len = len(training_data)
# Using K-fold
kFold = KFold(training_data_len, n_folds=10)

exitingResults = []

curInd = 0

for trainId, testId in kFold:
    x_model, x_test_model = X.values[trainId], X.values[testId]
    y_model, y_test_model = Y.values[trainId], Y.values[testId]
    # Fitting kNN Classifier to the data
    knnClf.fit(x_model, y_model)
    predictions = knnClf.predict(x_test_model)

    accuracy = accuracy_score(y_test_model, predictions)
    exitingResults.append(accuracy)
    curInd += 1
    print("Fold ID : %d Accuracy : %.6f" % (curInd, accuracy))   
    
mean_outcome = np.mean(exitingResults)
print("Mean Accuracy : %.6f" %mean_outcome ) 


# ### 4.3 Random Forest Classifier

# Random Forest is another classifier which use classification algorithm it is the forest decision trees, it reduces correlation between the trees and it use a random sample of features.

# ### 4.3.1 Random Forest Classifier (Testing Data)

# In[22]:


# Chossnig Random Forest Classifier

randomClf = RandomForestClassifier()

# We pick some combanitaions of parameter to try

parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Fitting Random Forest Classifier to the data
randomClf.fit(x_data, y_data)

ranClfPred = randomClf.predict(x_test_data)
print(accuracy_score(y_test_data, ranClfPred))


# ### 4.3.2 Random Forest Classifier (Training Data)

# In[23]:


# Chossnig Random Forest Classifier
training_data_len = len(training_data)
# Using K-fold
kFold = KFold(training_data_len, n_folds=10)

exitingResults = []

curInd = 0

for trainId, testId in kFold:
    x_model, x_test_model = X.values[trainId], X.values[testId]
    y_model, y_test_model = Y.values[trainId], Y.values[testId]
    # Fitting Random Forest Classifier to the data
    randomClf.fit(x_model, y_model)
    ranClfPred = randomClf.predict(x_test_model)

    accuracy = accuracy_score(y_test_model, ranClfPred)
    exitingResults.append(accuracy)
    curInd += 1
    print("Fold ID : %d Accuracy : %.6f" % (curInd, accuracy))   
    
mean_outcome = np.mean(exitingResults)
print("Mean Accuracy : %.6f" % mean_outcome ) 


# ### 4.4 Decision Tree Classifier

# Decision Tree Classifier has one of the best predictive feature to split the data. This algorithm can handle numerical and categorical data, and missing data. Furthermore, this algorithm can determine the most important features and can handle large data set. 
# Everything algorithm has its own limitation, one of the main cons and limitation of DTC is, we cannot model the interaction between the features, and the data can easily be over fit or under fit.

# ### 4.4.1 Decision Tree Classifier (Testing Data)

# In[24]:


# Chossnig Decision Tree Classifier
treeClf = DecisionTreeClassifier()
# Fitting Decision Tree Classifier to the data
treeClf.fit(x_data,y_data)

treeClfPred = treeClf.predict(x_test_data)
print(accuracy_score(y_test_data, treeClfPred))


# ### 4.4.2 Deision Tree Classifier (Trianing Data)

# In[25]:


# Chossnig Decision Tree Classifier
training_data_len = len(training_data)
kFold = KFold(training_data_len, n_folds=10)

exitingResults = []

curInd = 0

for trainId, testId in kFold:
    x_model, x_test_model = X.values[trainId], X.values[testId]
    y_model, y_test_model = Y.values[trainId], Y.values[testId]
    # Fitting Decision Tree Classifier to the data
    treeClf.fit(x_model,y_model)
    treeClfPred = treeClf.predict(x_test_model)

    accuracy = accuracy_score(y_test_model, treeClfPred)
    exitingResults.append(accuracy)
    curInd += 1
    print("Fold ID : %d Accuracy : %.6f" % (curInd, accuracy))   
    
mean_outcome = np.mean(exitingResults)
print("Mean Accuracy : %.6f" % mean_outcome ) 


# ## 5.0 Model Tuning

# To Improve the performance of models, we can use for example gird search on chosen metrics. One of the well-known method for tuning is GridSearchCV. Using this method, we will be tuning the hyper parameters that would result in optimized models. This method works by building several models with all the parameters combinations specified, and it runs a default of 3 cross validations to return a set of parameters that has the highest accuracy score.
# In conclusion, we can get a higher accuracy in all of our classifiers  after we tune the hyper parameters.

# ### 5.1 Logistic Regression (Model Tuning)

# In[26]:


# Logistic Regression
logisticRegr = LogisticRegression(random_state=42)
# C values which they can be between small number like 0.001 up to 1000
c_val = np.logspace(0, 4, 10)
# Two penalty options
penalty = ['l1', 'l2']

log_tun_paramss = dict(C=c_val, penalty=penalty)

scorer = make_scorer(accuracy_score)

# Runing GridSearchCV for the Logistic Regression classifier
logisticReg_gridCv = GridSearchCV(logisticRegr, log_tun_paramss, scoring=scorer)
logisticReg_gridObj = logisticReg_gridCv.fit(x_data, y_data)

# Setting our classifier
logisticRegr = logisticReg_gridObj.best_estimator_

logisticRegr_tuned_pred = logisticRegr.predict(x_test_data)

lr = accuracy_score(y_test_data, logisticRegr_tuned_pred)
print(accuracy_score(y_test_data, logisticRegr_tuned_pred))


# Comparing the Model Tuning result of Logistic Regression to the Logistic Regration from part 4.1 we can notice that the accuracy has increased slightly after our hyper tunning.

# ### 5.2 kNN Classifier (Model Tunning) 

# In[27]:


# kNN Classifier
knnClf = KNeighborsClassifier()
knnClf_params = {"n_neighbors": np.arange(5, 35, 4), "metric": ["euclidean", "minkowski"]}
scorer = make_scorer(accuracy_score)

knnClf_gridCv = GridSearchCV(knnClf, knnClf_params, scoring=scorer)
knnClf_gridObj = knnClf_gridCv.fit(x_data, y_data)

# Setting parameters for kNN Classifier
knnClf = knnClf_gridObj.best_estimator_

knnClfPred = knnClf.predict(x_test_data)

knn = accuracy_score(y_test_data, knnClfPred)
knn


# Comparing the Model Tuning result of kNN Classifier to the kNN Classifier from part 4.2 we can notice that the accuracy has increased slightly using hyper parameter.

# ### 5.3 Random Forest Classifier (Model Tunning)

# In[28]:


# Random Forest Classifier
# Chossing the classifier
randClf = RandomForestClassifier()
# Chossing parameters
randClfParams = {'n_estimators': [4, 9, 15], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 5, 10,15], 
              'min_samples_split': [2, 3, 5,10],
              'min_samples_leaf': [1,5,8]
             }

scorer = make_scorer(accuracy_score)

randClfGrid = GridSearchCV(randClf, randClfParams, scoring=scorer)
randClfGridFit = randClfGrid.fit(x_data, y_data)

randClf = randClfGridFit.best_estimator_
randClfPred = randClf.predict(x_test_data)

rfc = accuracy_score(y_test_data, randClfPred)
rfc


# Comparing the Model Tuning result of Random Forest Classifier to the Random Forest Classifier from part 4.3 we can notice that the accuracy has increased in higher % compared to the other classifiers using hyper parameter.

# ### 5.4 Tree Classifier (Model Tunning)

# In[29]:


# Tree Classifier
# Chossing the model
treeClf = DecisionTreeClassifier()
treeClf.fit(x_data,y_data)
# chossing parametrs
treeClfParams = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10, 100,500, 1000],
              "max_leaf_nodes": [None, 5, 10, 20,25],
              }

scorer = make_scorer(accuracy_score)

treeClfGrid = GridSearchCV(treeClf, treeClfParams, scoring=scorer)
treeClfGridFit = treeClfGrid.fit(x_data, y_data)

treeClf = treeClfGridFit.best_estimator_
treeClfPred = treeClf.predict(x_test_data)

tc = accuracy_score(y_test_data, treeClfPred)
tc


# Comparing the Model Tuning result of Tree Classifier to the Tree Classifier from part 4.4 we can notice that the accuracy has increased using hyper parameter.

# In[30]:


# Store accuracy in a dataframe

Classifier = pd.DataFrame(columns = ["Classifier", "Accuracy"])
Classifier.loc[0,"Classifier"] = "Logistic Regression"
Classifier.loc[0,"Accuracy"] = lr
Classifier.loc[1,"Classifier"] = "kNN"
Classifier.loc[1,"Accuracy"] = knn
Classifier.loc[2,"Classifier"] = "Random Forest"
Classifier.loc[2,"Accuracy"] = rfc
Classifier.loc[3,"Classifier"] = "Tree Classifier"
Classifier.loc[3,"Accuracy"] = tc


Classifier


# According to out tunnig, the accuracy of all of the our classifier increased. So from the table above we can notice that Random Forest and Tree Classifier has the highet accuracy, but Random Forest is slightly higher which closer to 80%. There are several reasons that this classifier can have higher accuracy such as, 
# 
# 1. It runs efficiently on large data
# 2. It gives estimation of what variables are important
# 3. It generate an internal unbiased estimate of the generalization error as the forest forest bulding progress.
# 
# So wee pick this classifier in the next section. 

# ## 6.0 Testing

# In[31]:


# Scales all features
std_scaler = StandardScaler()

x_data_scaled = std_scaler.fit_transform(x_data)
x_test_scaled = std_scaler.fit_transform(x_test_data)

randClf = RandomForestClassifier()

randClfParams = {'n_estimators': [4, 9, 15], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 5, 10,15], 
              'min_samples_split': [2, 3, 5,10],
              'min_samples_leaf': [1,5,8]
             }

scorer = make_scorer(accuracy_score)

randClfGrid = GridSearchCV(randClf, randClfParams, scoring=scorer)
randClfGridFit = randClfGrid.fit(x_data_scaled, y_data)

randClf = randClfGridFit.best_estimator_
randClfPred = randClf.predict(x_test_scaled)

conMatrix = confusion_matrix(y_test_data, randClfPred)

print(accuracy_score(y_test_data, randClfPred))

print(conMatrix)

The score is 80% of the training data. We see that the model generalizes well as the accuracy doesn't drop a lot from the training to the testing set. Also, we know the model performing much better in the case of <50K (a12,a22,a32), since there are more samples to learn from, and progressively gets worse at predicting the upper-income brackets.
# # Thank You
