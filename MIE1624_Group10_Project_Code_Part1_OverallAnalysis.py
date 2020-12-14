#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Data science can be thought of as a study of mathematical algorithms to generate elegant and effective solutions to complex physics-based and financial problems. Due to the growing scope and demand for data analysts, scientists, and relevant business professionals, education in the field of data science has become a major area of focus by students, professionals, and entrepreneurs. 
# 
# Through this IPython notebook, we analyzed the current situation of the data science job market. We also focused on how formal education, especially, Master's programs (from both Technical and Business perspective) can help in developing a professional insight in data-based jobs. The important and basic qualities for a typical data scientist was  to improve the MIE 1624 course At the end we we also looked into different Technical and Managerial Data-based curriculums to design two comprehensive data-science curriculums as well as an EdTech program. 
# 
# Three different kinds of datasets were used in this analysis. 
# 1. The 2017 Kaggle Survey Data
# 2. Data Science-Jobs  Data from Indeed 
# 3. Technical and Business Data Science Master's Program Informations collected from different schools
# 
# In the beginning, some basic exploratory analysis of the Kaggle dataset will be discussed. Then, the current data science job conditions will be analyzed. At this point, Indeed webscrapping and relevant data analysis will also be discussed. At the end, the curriculum topics analysis and importance of MOOC and EdTech programs will be highlighted.

# In[1]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install wordcloud')
import numpy as np # for linear algebra operations
import pandas as pd # data processing in dataframe, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting data
import seaborn as sns # data visualization
sns.set(color_codes=True)

import operator

import plotly.offline as py #necessary for more data visualization
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from subprocess import check_output


# ## Load Kaggle data:
# 
# Using the kaggle platform an industry-wide comprehensive survey on data science and machine learning was conducted in 2017. The survey included both data science workers and learners and collected more than 16,000 responses. For evaluating and redesigning the 'MIE 1624: Introduction to Data Science and Analytics' course syllabus this dataset is analyzed.

# In[2]:


# load the multiple choice question survey response
mcdata = pd.read_csv('multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory = False)
# load the currency conversion rate data 
cvRates = pd.read_csv('conversionRates.csv', encoding="ISO-8859-1", low_memory = False)
mcdata.head()


# # Exploratory Analysis and Data Visualization:
# 
# First, the demographic properties of the respondants are visualized.

# In[3]:


# How diversified (in terms of Gender) is Data Science? 
sns.set(style = 'whitegrid', context = 'talk')
Fig_1 = plt.figure(figsize=(8,6))
gen_mcdata = mcdata['GenderSelect'].value_counts()
sns.barplot(y=gen_mcdata.index, x=gen_mcdata.values, alpha=0.8, palette = "gist_stern")
plt.yticks(range(len(mcdata['GenderSelect'].value_counts().index)), 
           ['Male', 'Female','Other','Non-confirming'])
plt.title("Gender distribution of Kaggle suervey participants", fontsize=18)
plt.xlabel("Number of participants", fontsize=16)
plt.ylabel("Gender", fontsize=16)
plt.show()
print('Proportion of women in this survey is only: {:0.1f}% '
      .format(100*len(mcdata[mcdata['GenderSelect']=='Female'])/len(mcdata['GenderSelect'].dropna())))
print('Proportion of men in this survey: {:0.1f}% '
      .format(100*len(mcdata[mcdata['GenderSelect']=='Male'])/len(mcdata['GenderSelect'].dropna())))


# It is apparent that the computing world, especially, data science is pretty unevenly distributed. Only approx. 17% of the respondants was women. 

# In[5]:


# How old are most of the data scientists?
from scipy.stats import norm

age_mcdata = mcdata[(mcdata['Age']>=10) & (mcdata['Age']<=70)]
age = age_mcdata['Age'].value_counts()
age_trace = go.Bar(x = age.index, y = age.values, marker = 
                   dict(color = age.values, colorscale = 'Portland', showscale = True))
age_layout = go.Layout(title = 'Age distribution of Kaggle survey participants',
                      yaxis = dict(title = 'Number of participants', zeroline = False),
                      xaxis = dict(title = 'Age', zeroline = False))
Fig_2 = go.Figure(data = [age_trace], layout = age_layout)
py.iplot(Fig_2, filename = 'Age_distribution')

(mu, sigma) = norm.fit(age_mcdata['Age'])
print('The average age of the survey participants is: {:0.2f}%'.format(mu))
print('And the variance is: {:0.2f}%'.format(sigma))


# The age analysis clearly shows that Data Science is dominated by young professionals. 

# In[6]:


# Where are Data Scientists from? 
Fig_3 = plt.figure(figsize=(12,16))
countries = mcdata['Country'].value_counts().head(25)
sns.barplot(y=countries.index, x=countries.values, alpha=0.85, palette = "gnuplot")
plt.title("Country-wise distribution of survey participants\n", fontsize=24)
plt.xlabel("Number of participants", fontsize=24)
#plt.ylabel("Country", fontsize=20)
plt.yticks(fontsize = 21)
plt.xticks(fontsize = 20)
plt.show()
print('Only {:0.2f}% of the instances are Canadians'.format(100*len(mcdata[mcdata['Country']=='Canada'])/len(mcdata)))


# It seems the biggest field of data science is our neighboring country, the U.S., whereas, we are only occupying 2.63% of the market. But, what portions of the respondants are actually professional data scientists and how many of them are are actually students?

# In[7]:


# Here the student density distribution is plotted
# plot the world map
world_map = pd.read_csv("location_map.csv")
map_df = pd.merge(mcdata[['Country','StudentStatus']], world_map,
                 left_on = 'Country', right_on = 'COUNTRY')
map_df = map_df.groupby(['Country','CODE'])['StudentStatus'].aggregate('count').reset_index()
map_df.columns = ['Country','Code','Count']

data = [dict(type = 'choropleth', locations = map_df.Code, z = map_df.Count,
            text = map_df.Country, colorscale = 'Portland', marker =
            dict(line = dict(color = 'rgb(150,150,150)', width = 0.5)),
            colorbar = dict(title = 'Count of students'))]
layout = dict(title = 'Country-wise student respondants', geo =
             dict(showframe = False, showcoastlines = True, projection =
                 dict(type = 'Robinson')))
Fig_4 = plt.figure(figsize = (8,14))
Fig_4 = dict(data = data, layout = layout)
py.iplot(Fig_4, validate = False, filename = 'Student_world_map')


# In[8]:


# Worker density distribution on a world map
world_map = pd.read_csv("location_map.csv")
map_df = pd.merge(mcdata[['Country','EmploymentStatus']], world_map,
                 left_on = 'Country', right_on = 'COUNTRY')
map_df = map_df.groupby(['Country','CODE'])['EmploymentStatus'].aggregate('count').reset_index()
map_df.columns = ['Country','Code','Count']

data = [dict(type = 'choropleth', locations = map_df.Code, z = (map_df.Count/map_df.Count.sum())*100,
            text = map_df.Country, colorscale = 'Portland', marker =
            dict(line = dict(color = 'rgb(150,150,150)', width = 0.5)),
            colorbar = dict(title = '% of job'))]
layout = dict(title = 'Country-wise employee respondants', 
              geo = dict(showframe = False, showcoastlines = True, type = 'Robinson'))
Fig_5 = plt.figure(figsize = (8,14))
Fig_5 = dict(data = data, layout = layout)
py.iplot(Fig_5, validate = False, filename = 'Employee_world_map')


# ### Necessity of Formal Education
# 
# In the Kaggle survey, necessity of formal education was discussed. The formal education of the the survey respondants were analyzed and it's conspicuous that majority of the Data Scientists are university graduates. In fact, it is seen that approx. 42% of the respondants had a Master's degree while approx. 16% hold a Ph.D. In later sections, the success rate of these data scientists 
# 
# The undergraduate major of these participants were also looked into and it seems that although a lot of respondants came from a Computer Science background, a significant portion of them are from non-computer focused group, Mathematics, and Physics. Even a lot of responses came from Social Science, Humanities, Management Information Systems as well as Fine Arts field. So, its a mythbuster that data science is not only the field of computer scientists anymore.   

# In[10]:


# Formal education plot of the respondants
mcdata['FormalEducation']=mcdata['FormalEducation'].replace(to_replace =
                            'Some college/university study without earning a bachelor\'s degree',
                            value = 'Some college/university-scale study')
mcdata['FormalEducation']=mcdata['FormalEducation'].replace(to_replace =
                            'I did not complete any formal education past high school',
                            value = 'High school only')
mcdata['FormalEducation']=mcdata['FormalEducation'].replace(to_replace =
                            'I prefer not to answer',
                            value = 'No answer')

edu = mcdata['FormalEducation'].value_counts()
labels = (np.array(edu.index))
values = (np.array((edu/edu.sum())*100))
colors = ['rgb(155,206,227)', 'rgb(31,120,180)', 'rgb(178,223,138)', 'rgb(51,160,44)', 
          'rgb(251,154,153)', 'rgb(227,26,28)', 'rgb(240,220,120)']
edu_trace = go.Pie(labels=labels, values=values, hoverinfo='label+percent', textfont=dict(size=20, color = 'black'), 
                   showlegend=True, marker=dict(colors=colors, line=dict(color='#000000', width=0.5)))

edu_layout = go.Layout(title='<b>Formal Education of the Kaggle survey participants</b>',
                       legend = dict(x = 0.85, y = 0.9, traceorder = 'normal', orientation = 'v'))
Fig_6 = go.Figure(data=[edu_trace], layout=edu_layout)
py.iplot(Fig_6, filename="Formal_Education")

# What was the undergraduate major of these participants?

mcdata['MajorSelect']=mcdata['MajorSelect'].replace(to_replace =
                            'Information technology, networking, or system administration',
                            value = 'Information tech/System admin')
mcdata['MajorSelect']=mcdata['MajorSelect'].replace(to_replace =
                            'A social science', value = 'Social Science')
mcdata['MajorSelect']=mcdata['MajorSelect'].replace(to_replace =
                            'A humanities discipline', value = 'Humanities Discipline')
mcdata['MajorSelect']=mcdata['MajorSelect'].replace(to_replace =
                            'A health science', value = 'Health Science')
mcdata['MajorSelect']=mcdata['MajorSelect'].replace(to_replace =
                            'I never declared a major', value = 'No major')
majors = mcdata['MajorSelect'].value_counts()
Fig_7 = plt.figure(figsize=(6,6))
Fig_7 = sns.barplot(y=majors.index, x=majors.values, alpha=0.95, edgecolor = 'black', palette = "RdBu_r")
plt.title("Undergraduate Majors of the survey participants", fontsize=18, color = 'black')
plt.xlabel("Number of participants", fontsize=16, color = 'black')
#plt.ylabel("Undergraduate Major", fontsize=16)
plt.show()


# Then, we looked into the respondant's (already employed as a data scientist) profile, especially, those who claimed that they are a perfect fit for their job. Majority of those respondants mentioned that university education is very important to have a good career in data science.

# In[11]:


mcdata['UniversityImportance']=mcdata['UniversityImportance'].replace(to_replace ='N/A, I did not receive any formal education',
                                                                        value = 'N/A')
perfectshot = mcdata.loc[mcdata['TitleFit'] == 'Perfectly']
univimp=perfectshot['UniversityImportance'].value_counts()
labels = (np.array(univimp.index))
values = (np.array(univimp.values))
#colors2 = ['rgb(240,176,150)', 'rgb(131,220,180)', 'rgb(178,223,108)', 'rgb(155,206,227)','rgb(240,220,120)', 
           #'rgb(227,26,28)','rgb(240,210,128)']
trace = go.Pie(labels=labels, values=values,hoverinfo='label+percent', textfont=dict(size=20, color = 'black'), 
               showlegend=True, marker=dict(line=dict(color='#000000', width=0.5)))

layout = go.Layout(title='<b>Is University Education Important?</b>', legend = 
                   dict(x = .75, y = 0.9, traceorder = 'normal', orientation = 'v'))

data_trace = [trace]
Fig_8 = go.Figure(data=data_trace, layout=layout)
py.iplot(Fig_8, filename="University_Education_Requirement")


# The data science job opportunities, necessary libraries, and algorithms were also looked into. Here is a quick overview of the major data science job titles, libraries, and algorithms in practice. 

# In[12]:


import wordcloud
from wordcloud import WordCloud
# load the multiple choice question survey response
free_resp = pd.read_csv('freeformResponses.csv', encoding="ISO-8859-1", low_memory = False)

currjob_free = free_resp[pd.notnull(free_resp['CurrentJobTitleFreeForm'])]['CurrentJobTitleFreeForm'].str.lower()
currjob_free = currjob_free.str.replace(';',',')
currjob_free = currjob_free.str.replace('none',' ')
currjob_free = currjob_free.str.replace(' ','')
currjob_free = currjob_free.str.replace('notapplicable',' ')
currjob_text = currjob_free.str.cat(sep=',')
wordcld_currjob = WordCloud(collocations=False,height=800, width=600,  
                            max_words = 200, max_font_size = 400, background_color = 'black',
                            relative_scaling=0.2,random_state=100).generate((currjob_text))
#worklib_text
worklib_free = free_resp[pd.notnull(free_resp['WorkLibrariesFreeForm'])]['WorkLibrariesFreeForm'].str.lower()
worklib_free = worklib_free.str.replace(';',',')
worklib_free = worklib_free.str.replace('\n',',')
worklib_free = worklib_free.str.replace(' ','')
worklib_text = worklib_free.str.cat(sep=',')
wordcld_worklib = WordCloud(collocations=False,height=800, width=600,  
                            max_words = 200, max_font_size = 400, background_color = 'coral',
                            relative_scaling=0.2,random_state=100).generate((worklib_text))
#worklib_text
workalg_free = free_resp[pd.notnull(free_resp['WorkAlgorithmsFreeForm'])]['WorkAlgorithmsFreeForm'].str.lower()
workalg_free = workalg_free.str.replace(';',',')
workalg_free = workalg_free.str.replace('none',' ')
workalg_free = workalg_free.str.replace(' ','')
workalg_free = workalg_free.str.replace('notapplicable',' ')
workalg_text = workalg_free.str.cat(sep=',')
wordcld_workalg = WordCloud(collocations=False,height=800, width=600,  
                            max_words = 250, max_font_size = 400, background_color = 'lightyellow',
                            relative_scaling=0.2,random_state=100).generate((workalg_text))
#worklib_text
Fig_9 = plt.figure(figsize=(20,9))
Fig_9.add_subplot(1,3, 1)
plt.imshow(wordcld_currjob)
plt.axis('off')
plt.title('Data Science Jobs', fontsize = '18', color = 'black')
Fig_9.add_subplot(1,3, 2)
plt.imshow(wordcld_worklib)
plt.axis('off')
plt.title('Libraries You Need', fontsize = '18', color = 'black')
Fig_9.add_subplot(1,3, 3)
plt.imshow(wordcld_workalg)
plt.axis('off')
plt.title('Algorithms You Should Learn', fontsize = '18', color = 'black')

plt.show()


# Details of the data science job market with an added focus on current Canadian job situation will be discussed in the following section.

# # Job Market Analysis

# In this section, we'll try to answer questions like:
# - Which jobs are more frequent?
# - What are the highest paid jobs?
# - What skills do you need to secure good jobs?
# - Can machine learning algorithm help you to detect important features to get a highly paid job in your Data Science Career?
# - What's the current situation of the Canadian Data Science job market? How can the universities play a major role?

# In[13]:


# Which jobs are more frequent?

jobs = mcdata['CurrentJobTitleSelect']
jobs1 = jobs[~(mcdata['CurrentJobTitleSelect'] == 'Other')]
job_list = []
for i in jobs1.dropna():
    job_list.append(i)
# Selecting the genre of the data science jobs people are doing
job = pd.Series(job_list).value_counts().sort_values(ascending = True)[:25].to_frame()  

job_trace = go.Bar(y = job.index, x = job[0], marker = 
                      dict(color = job[0], colorscale = 'Portland', 
                           showscale = False), opacity = 0.9, orientation = 'h')
job_layout = go.Layout(title = '<b>The 2017 Data Science Job Market</b>',
                        yaxis = dict(zeroline = False,tickfont= dict(family ='Arial, sans-serif', size = 25, color = 'black')),
                        xaxis = dict(title = 'Job count',titlefont= dict(family ='Arial, sans-serif', size = 24, color = 'black'), 
                                     showticklabels = True, tickangle = 0, tickfont= dict(family ='Arial, sans-serif', size = 25, color = 'black')),
                       autosize = False, width = 1200, height = 1000, margin=go.Margin(l=450,r=10,b=100, t=100,pad=4))
Fig_10 = go.Figure(data = [job_trace], layout = job_layout)
py.iplot(Fig_10, filename = 'job_distribution')


# In[14]:


# But who gets the highest salary?
mcdata['CompensationAmount']=mcdata['CompensationAmount'].str.replace(',','')
mcdata['CompensationAmount']=mcdata['CompensationAmount'].str.replace('-','')
salary=mcdata[['EmploymentStatus','CompensationAmount','CompensationCurrency','Country','FormalEducation',
               'CurrentJobTitleSelect', 'Age','EmployerIndustry','EmployerSize','EmployerMLTime','EmployerSearchMethod']].dropna()
# load the currency conversion rate data 
cvRates = pd.read_csv('conversionRates.csv', encoding="ISO-8859-1", low_memory = False)
cvRates.drop('Unnamed: 0', axis=1,inplace=True)

salary=salary.merge(cvRates, left_on='CompensationCurrency', right_on='originCountry', how='left')
salary['Salary']= pd.to_numeric(salary['CompensationAmount'])*salary['exchangeRate']
Salary_up = salary[~(salary['CurrentJobTitleSelect'] == 'Other')]
salary_job = Salary_up.groupby('CurrentJobTitleSelect')['Salary'].median().to_frame().sort_values(by='Salary', ascending = True)[:25]

sal_trace = go.Bar(y = salary_job.index, x = salary_job.Salary, marker = 
                      dict(color = salary_job.Salary, colorscale = 'Portland', 
                           showscale = False), opacity = 0.9, orientation = 'h')
sal_layout = go.Layout(title = '<b> Compensation Amount (Median Value) by Data Scince Job Titles</b>',
                        yaxis = dict(zeroline = False,tickfont= dict(family ='Arial, sans-serif', size = 25, color = 'black')),
                        xaxis = dict(title = 'Salary ($)',titlefont= dict(family ='Arial, sans-serif', size = 24, color = 'black'), 
                                     showticklabels = True, tickangle = 0, tickfont= dict(family ='Arial, sans-serif', size = 25, color = 'black')),
                       autosize = False, width = 1200, height = 1000, margin=go.Margin(l=450,r=10,b=100, t=100,pad=4))
Fig_11 = go.Figure(data = [sal_trace], layout = sal_layout)
py.iplot(Fig_11, filename = 'job_salary_distribution')


# It is interesting to note that although the availibility of Operations Research Practitioner is less, the salary these people get is huge, compared to other genres. Research positions are relatively less frequent and less paid than their industrial/corporate counterparts. Another interesting observation is although we've seen earlier that many data scientists came from a computer science background, job market and salary for computer scientists are significantly less than the other genres. 
# 
# How about education? Does higher degrees matter?

# In[15]:


Salary_1 = salary[~(salary['FormalEducation'] == 'I prefer not to answer')]
Salary_1 = salary[~(salary['FormalEducation'] == 'No answer')]
salary_edu = Salary_1.groupby('FormalEducation')['Salary'].median().to_frame().sort_values(by='Salary', ascending = False)[:15]
salary_edu = salary_edu[pd.notnull(salary_edu.Salary)]
sal_edu_trace = go.Bar(x = salary_edu.index, y = salary_edu.Salary, marker = 
                      dict(color = salary_edu.Salary, colorscale = 'Viridis', line=dict(color='#000000', width=1), 
                           showscale = False), opacity = 0.95)
sal_edu_layout = go.Layout(title = '<b>Salary by Formal Education</b>',
                        yaxis = dict(title = 'Salary ($)', zeroline = False),
                        xaxis = dict(showticklabels = True, tickangle = 30, tickfont= dict(
                        family ='Arial, sans-serif', size = 12, color = 'black')),
                        autosize = False, width = 600, height = 700,margin=go.Margin(l=50,r=100,b=200,t=100,pad=4))
Fig_12 = go.Figure(data = [sal_edu_trace], layout = sal_edu_layout)
py.iplot(Fig_12, filename = 'salary_education')


# Yes, it is. The compensation rises significantly after Master's or Doctoral degrees.
# 
# At this point, the highest salary based job providers were searched for and not surprisingly, security, insurance, Enertainment, and Retail came within the top 5 of that list. Academics were at the bottom position. 

# In[16]:


# Which industries pay the most?
salary_emp = salary.groupby('EmployerIndustry')['Salary'].median().to_frame().sort_values(by='Salary', ascending = False)[:25]
emp_trace = go.Bar(x = salary_emp.index, y = salary_emp.Salary, marker = 
                      dict(color = salary_emp.Salary, colorscale = 'Jet', line=dict(color='#000000', width=1), 
                           showscale = False), opacity = 0.95)
emp_layout = go.Layout(title = '<b>Which industry pay the most</b>?',
                        yaxis = dict(title = 'Salary ($)', zeroline = False),
                        xaxis = dict(showticklabels = True, tickangle = 30, tickfont= dict(
                        family ='Arial, sans-serif', size = 14, color = 'black')),
                        autosize = False, width = 600, height = 750,margin=go.Margin(l=50,r=100,b=250,t=50,pad=4))
Fig_13 = go.Figure(data = [emp_trace], layout = emp_layout)
py.iplot(Fig_13, filename = 'job industry by salary')


# At this point, we wanted to analyze the data to further detect hidden features in the dataset upon which the salary depends. At first, we employed the random forest classifier to detect the top 15 features in the dataset. 

# In[17]:


tot_data=mcdata.merge(cvRates, left_on='CompensationCurrency', right_on='originCountry', how='left')
tot_data['Salary']= pd.to_numeric(tot_data['CompensationAmount'])*tot_data['exchangeRate']
tot_data = tot_data[tot_data['Salary'].notnull()]
tot_data.head()


# In[18]:


print('The median salary of data scientists (from the survey data) is : %0.2f'% np.median(tot_data.Salary) )


# In[19]:


tot_data = tot_data.drop('CompensationAmount',1)
tot_data = tot_data.drop('originCountry',1)
tot_data = tot_data.drop('exchangeRate',1)
y = tot_data['Salary']
X = tot_data.drop('Salary', 1)
X = X.fillna(0)
X_dummy = pd.get_dummies(X)
# created a binary salary group based on the median salary of the participants in the dataset
y_binary = y > 53812.17


# In[20]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_binary, test_size=0.3, random_state = 50)
classifier = RandomForestClassifier(n_estimators = 20, random_state = 50)
randfor_model = classifier.fit(X_train, y_train)
randfor_score = randfor_model.score(X_test, y_test)
print("The accuracy score is: ", randfor_score)


# In[21]:


randfor_output = pd.DataFrame()
randfor_output['FeatureName'] = X_dummy.columns.tolist()
randfor_output['FeatureImportance'] = randfor_model.feature_importances_
top15Features = randfor_output.sort_values("FeatureImportance", ascending = False)[:15]


# In[22]:


trace = go.Scatter(x=top15Features.FeatureName,y=top15Features.FeatureImportance,mode='markers',marker=dict(
                    sizemode = 'diameter',line=dict(color='#000000', width=0.5),size = 3000*top15Features.FeatureImportance, color = top15Features.FeatureImportance, 
                    colorscale='YlGnBu', opacity = 0.8, showscale=True))

layout = go.Layout(title='<b>Important features for High Salary </b>', autosize=False, width=800, height=800,
                   yaxis = dict(title = 'Feature Importance', titlefont = dict(family='Arial, sans-serif',size = 18), 
                                zeroline = False, tickfont= dict(family ='Arial, sans-serif', size = 16, color = 'black')),
                   xaxis = dict(showticklabels = True, tickangle = 30, tickfont= dict(
                        family ='Arial, sans-serif', size = 15, color = 'black')),
                   margin=go.Margin(l=150,r=150,b=250,t=100,pad=5))
data = [trace]
Fig_14 = go.Figure(data=data, layout=layout)
py.iplot(Fig_14, filename="random forest topfeature")


# It is interesting to note that demographic features, age, and learning catergories showed up as important feature paratmeters alongside job experience and computer science major as well as major job challenges. However, this analysis doesn't clearly show the positive and negative association of these features with salary. So, we did a logistic regression as well, and calculated the coefficients.

# In[23]:


top50Features = randfor_output.sort_values("FeatureImportance", ascending = False)[:50]
X_LR = X_dummy[top50Features['FeatureName']]

from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X_LR, y_binary, test_size = 0.3, random_state=50)
classifier_LR = LogisticRegression()
model_LR =classifier_LR.fit(X_train, y_train)
model_LR_score = model_LR.score(X_test, y_test)

print("The accuracy score from Logistic Regression is: ", model_LR_score)


# In[24]:


LR_output = pd.DataFrame()
LR_output['FeatureName'] = X_LR.columns.tolist()
LR_output['FeatureCoefficients'] = model_LR.coef_[0].tolist()
LR_output.head(10)


# In[25]:


coefs = LR_output.FeatureCoefficients.ravel()
pos_coefs = np.argsort(coefs)[-10:]
neg_coefs = np.argsort(coefs)[:10]
top_coefs = np.hstack([neg_coefs,pos_coefs])


# In[26]:


trace = go.Scatter(x=np.array(LR_output.FeatureName)[top_coefs],y=coefs[top_coefs],mode='markers',marker=dict(symbol = 'triangle-up',
                    sizemode = 'diameter',line=dict(color='#000000', width=0.5),size = 50*abs(coefs[top_coefs]), color = coefs[top_coefs], 
                    colorscale='YlGnBu', opacity = 0.8, showscale=True))

layout = go.Layout(title='<b>Important features in Data Science</b>', autosize=False, width=850, height=800,margin=go.Margin(l=100,r=250,b=250,t=80,pad=5),
                   yaxis = dict(title = 'Feature Importance', titlefont = dict(family='Arial, sans-serif',size = 18), 
                                zeroline = False, tickfont= dict(family ='Arial, sans-serif', size = 15, color = 'black')),
                   xaxis = dict(zeroline = False,tickfont= dict(family ='Arial, sans-serif', size = 12, color = 'black'), tickangle = 35))

data = [trace]
Fig_15 = go.Figure(data=data, layout=layout)
py.iplot(Fig_15, filename="logistic feature")


# So, it seems that important features for highly paid data scientist jobs can be identified using machine learning technique. This is more meaningful than the previous random forest classification. Only one demographic feature (the most important one though), the country USA as a whole, showed positive association to secure a highly paid data analyst job. It's not surprising based on their current economy, technical advancement, job-market for data science and high supply of data-science graduates fueling that market. Beside this, job experiences, industry size, formal education and high degree, knowing Python are the top features for securing a good job in data science. 

# In[27]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test.ravel(), model_LR.decision_function(X_test))
roc_auc = auc(fpr, tpr)
sns.set_style('white')
Fig_16 = plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, color='red', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate', fontsize = 16)
plt.legend(loc="lower right", fontsize = 14)
plt.show()


# To check the performance of our model, the ROC curve is plotted above, which indicates a good overall model performance with an area under the curve value of 0.90.

# ### Indeed Job Analysis
# At this point the Canadian Job Market will be presented from Indeed analysis. The Indeed webscrapping code (MIE1624_Group10_ProjectCode_Part2_IndeedWebScrapping) is uploaded separately to reduce runtime of this IPython script. In that particular code we looked into major US and Canada cities for Data Scientist jobs. The outputs were saved into two csv files - Canada_Indeed and USA_Indeed - and loaded into this dataframe to look into the key qualities employers are searching among the data scientists. 

# In[28]:


# load the Canada Indeed file
Can_indeed = pd.read_csv('Canada_Indeed.csv', encoding="ISO-8859-1", low_memory = False)
# load the US Indeed file
US_indeed = pd.read_csv('USA_Indeed.csv', encoding="ISO-8859-1", low_memory = False)


# In[29]:


# plotting using plotly
can_skill_trace = go.Bar(x=Can_indeed.Term, y=Can_indeed.NumPostings, marker=dict(
                        color=Can_indeed.NumPostings, colorscale = 'Portland', reversescale = False),
                        opacity = 0.95, width = 0.4)

can_skill_layout = go.Layout(title='<b>What skills do the Canadian industries looking for?</b>', 
                       yaxis = dict(title = 'Percentage Appearing in Job Ads', zeroline = False),
                       xaxis = dict(showticklabels = True, tickangle = -45, tickfont= dict(
                       family ='Arial, sans-serif', size = 10, color = 'black')),
                       autosize = False, width = 1000, height = 700,margin=go.Margin(l=100,r=100,b=150, t=60,pad=4))

Fig_29 = go.Figure(data=[can_skill_trace], layout=can_skill_layout)
py.iplot(Fig_29, filename="Skill_set_Canada")

# plotting using plotly
US_skill_trace = go.Bar(x=US_indeed.Term, y=US_indeed.NumPostings, marker=dict(
                        color=US_indeed.NumPostings, colorscale = 'Portland', reversescale = False),
                        opacity = 0.95, width = 0.4)

US_skill_layout = go.Layout(title='<b>What skills do the USA industries looking for?</b>', 
                       yaxis = dict(title = 'Percentage Appearing in Job Ads', zeroline = False),
                       xaxis = dict(showticklabels = True, tickangle = -45, tickfont= dict(
                       family ='Arial, sans-serif', size = 10, color = 'black')),
                       autosize = False, width = 1000, height = 700,margin=go.Margin(l=100,r=100,b=150, t=60,pad=4))

Fig_30 = go.Figure(data=[US_skill_trace], layout=US_skill_layout)
py.iplot(Fig_30, filename="Skill_set_USA")


# It is interesting to note that, in Canada, some of the soft skills, i.e., communication, strategic thinking, planning, leadership, etc. get high orsimilar priority as technical skills or programming background. On the other hand, USA jobs focus more on programming and other technical aspects. The soft skills are important in the US, however, it's not as significant as we observed in the Canadian Job Market. This observation state the fact that the US job market has more research and technical positions to offer to data scientists than Canada, and also states the need for strong business focused data science university degrees for the Canadian job market.

# # How to become a good Data Scientist?

# Data science is the study and application of algorithms. So, to become a good data scientist, it is imperative to have a good understanding of the key machine learning techniques as well as skills in solving different kinds of problem.

# ### Important Machine Learning Skills:

# In[30]:


mcdata['MLSkillsSelect']=mcdata['MLSkillsSelect'].replace(to_replace =
                            'Other (please specify; separate by semi-colon)',
                            value = 'Other')


# In[31]:


skills = mcdata['MLSkillsSelect'].str.split(',')

skills_list = []
for i in skills.dropna():
    skills_list.extend(i)
skill = pd.Series(skills_list).value_counts().sort_values(ascending = False)[:12].to_frame()  

skills_trace = go.Bar(x = skill.index, y = skill[0], marker = 
                      dict(color = skill[0], colorscale = 'Portland', 
                           showscale = False), opacity = 0.8, width = 0.7)
skills_layout = go.Layout(title = '<b>Which Machine Learning Skills Do You Need Most?</b>',
                        yaxis = dict(title = 'Count', zeroline = False,tickfont= dict(
                        family ='Arial, sans-serif', size = 16, color = 'black')),
                        xaxis = dict(showticklabels = True, tickangle = 30, tickfont= dict(
                        family ='Arial, sans-serif', size = 16, color = 'black')),
                        autosize = False, width = 700, height = 700,margin=go.Margin(l=100,r=100,b=250, t=60,pad=4))
Fig_17 = go.Figure(data = [skills_trace], layout = skills_layout)
py.iplot(Fig_17, filename = 'skills_distribution')


# ### Important Machine Learning Techniques to Learn:

# In[32]:


tools = mcdata['MLTechniquesSelect'].str.split(',')
tools_list = []
for i in tools.dropna():
    tools_list.extend(i)
tool = pd.Series(tools_list).value_counts().sort_values(ascending = False)[:12].to_frame()  

tools_trace = go.Bar(x = tool.index, y = tool[0], marker = 
                      dict(color = tool[0], colorscale = 'Portland', 
                           showscale = False), opacity = 0.8, width = 0.7)
tools_layout = go.Layout(title = '<b>Top Machine Learning Techniques You Need</b>',
                        yaxis = dict(title = 'Count', zeroline = False,tickfont= dict(
                        family ='Arial, sans-serif', size = 16, color = 'black')),
                        xaxis = dict(showticklabels = True, tickangle = 30, tickfont= dict(
                        family ='Arial, sans-serif', size = 16, color = 'black')),
                        autosize = False, width = 700, height = 700,margin=go.Margin(l=100,r=150,b=250, t=60,pad=4))
Fig_18 = go.Figure(data = [tools_trace], layout = tools_layout)
py.iplot(Fig_18, filename = 'tools_distribution')


# ### Your Choices of Programming Language

# In[33]:


perfectshot = mcdata.loc[mcdata['TitleFit'] == 'Perfectly']
prog_lang = perfectshot['LanguageRecommendationSelect'].value_counts()
prog_labels = (np.array(prog_lang.index))
prog_values = (np.array(prog_lang.values))

prog_trace = go.Pie(labels=prog_labels, values=prog_values,hoverinfo='label+percent', textfont=dict(size=20, color = 'black'), 
                   showlegend=True, marker=dict(line=dict(color='#000000', width=0.5)))

prog_layout = go.Layout(title='<b>Which Languages Do High-Profile Professionals Recommend?</b>', legend = 
                       dict(x = .95, y = 0.95, traceorder = 'normal', orientation = 'v'),
                       autosize = False, width = 700, height = 600,margin=go.Margin(l=100,r=50,b=150, t=60,pad=4))

data_trace = [prog_trace]
Fig_19 = go.Figure(data=data_trace, layout=prog_layout)
py.iplot(Fig_19, filename="Language Suggestion")


# In[34]:


codewriter = mcdata.loc[mcdata['CodeWriter'] == 'Yes']
prog_lang = codewriter['LanguageRecommendationSelect'].value_counts()
prog_labels = (np.array(prog_lang.index))
prog_values = (np.array(prog_lang.values))

prog_trace = go.Pie(labels=prog_labels, values=prog_values,hoverinfo='label+percent', textfont=dict(size=20, color = 'black'), 
                   showlegend=True, marker=dict(line=dict(color='#000000', width=0.5)))

prog_layout = go.Layout(title='<b>Which Languages Do Code-Writers Recommend?</b>', legend = 
                       dict(x = .95, y = 0.95, traceorder = 'normal', orientation = 'v'),
                       autosize = False, width = 700, height = 600,margin=go.Margin(l=100,r=50,b=150, t=60,pad=4))

data_trace = [prog_trace]
Fig_20 = go.Figure(data=data_trace, layout=prog_layout)
py.iplot(Fig_20, filename="Language Suggestion 2")


# It is apparent that from both Code-Writers and Superior Professionals perspective, Python and R are the most important programming language tools. But  how do their distribution look like based on professional data analyst's usuage perspective?

# In[35]:


Py_R = tot_data[["WorkToolsFrequencyR","WorkToolsFrequencyPython"]].fillna(0)
Py_R.replace(to_replace=['Rarely','Sometimes','Often','Most of the time'], value=[1,2,3,4], inplace=True)
Py_R['PythonVsR'] = ['R' if (freq1 >2 and freq1 > freq2) else
                    'Python' if (freq1<freq2 and freq2>2) else
                    'Both' if (freq1==freq2 and freq1 >2) else
                    'None' for (freq1,freq2) in zip(Py_R["WorkToolsFrequencyR"],Py_R["WorkToolsFrequencyPython"])]
tot_data['PythonVsR']=Py_R['PythonVsR']

lan_data = tot_data[tot_data['PythonVsR']!='None']

Py_n_R = lan_data.groupby(['CurrentJobTitleSelect','PythonVsR'])['Age'].count().to_frame().reset_index()
Py_n_R.pivot('CurrentJobTitleSelect','PythonVsR','Age').plot.barh(width = 0.95, fontsize = 14,colormap='Set1', 
                                                                  edgecolor = 'black', alpha = 0.95)
Fig_21 = plt.gcf()
ax = plt.gca()
Fig_21.set_size_inches(8,8)
ax.set_xlabel('# of Participants', fontsize = 14)
ax.set_ylabel('')
#plt.savefig("PyvsR.png", dpi=300)
plt.show()


# It seems that Python has higher importance than R for most of the job fields. Exceptions are Statisticians, Data Analyst, and Business Analyst positions. 

# # Curriculum Analysis of Other Schools and New Curriculum Design

# Based on the above mentioned ideas, the course syllabus for the MIE 1624 can be redesigned focusing on:
# - Important Machine Learning Techiques
# - Important Machine Learning Skills
# - Important Programming Languages and Libraries to Learn

# Also as a part of the course project, we looked into several schools that offer Technical and Business Master's programs for students and analyzed their course curriculums, additional project availability, and entry requirements.

# Load the Managerial Courses Data

# In[36]:


business_curr = pd.read_csv('Managerial_Business_Data_updated.csv', encoding="ISO-8859-1", low_memory = False)
business_curr.head()


# The major topics covered through the fundamental courses of 10 famous schools offering a Managerial and Business Data Science Master's program have been analyzed and plotted in the section below. Also, important admission requirements for these schools are highlighted.

# In[37]:


# Identifying and counting the recurrance of 
# fundamental courses in different schools
Fun_courses = business_curr['Fundamental Courses'].str.split(',')
FC_list = []
for x in Fun_courses.dropna():
    FC_list.extend(x)
Fundamentals = (pd.Series(FC_list).value_counts()/10*100).sort_values(ascending = False)[:20].to_frame()  

# Identifying and counting the application requirements
App_req = business_curr['Application Requirements'].str.split(',')
AR_list = []
for x in App_req.dropna():
    AR_list.extend(x)
Requirements = (pd.Series(AR_list).value_counts()/10*100).sort_values(ascending = False)[:20].to_frame()  

sns.set_style('whitegrid')

Fig_22 = plt.figure(figsize=(6,7))
Fig_22 = sns.barplot(x=Fundamentals[0], y= Fundamentals.index, alpha=0.8, 
                    palette = "terrain")
plt.title("Which courses are necessary?", fontsize=18)
plt.xlabel("Importance (%)", fontsize=16)
plt.gca().set_yticklabels(Fundamentals.index, fontsize = 14, rotation = 0)
plt.show()
Fig_23 = plt.figure(figsize=(6,7))
Fig_23 = sns.barplot(x=Requirements[0], y= Requirements.index, alpha=0.8, 
                    palette = "afmhot")
plt.title("Application Requirements", fontsize=18)
plt.xlabel("Importance (%)", fontsize=16)
plt.gca().set_yticklabels(Requirements.index, fontsize = 14, rotation = 0)
plt.show()


# Load the Technical Courses Data

# In[38]:


technical_curr = pd.read_csv('Technical_Curriculam_Data_updated.csv', encoding="ISO-8859-1", low_memory = False)
technical_curr.head()


# In[39]:


# Identifying and counting the recurrance of 
# fundamental courses in different schools
Fun_courses = technical_curr['Fundamental Courses'].str.split(',')
FC_list = []
for x in Fun_courses.dropna():
    FC_list.extend(x)
Fundamentals = (pd.Series(FC_list).value_counts()/10*100).sort_values(ascending = False)[:20].to_frame()  

# Identifying and counting the application requirements
App_req = technical_curr['Application Requirements'].str.split(',')
AR_list = []
for x in App_req.dropna():
    AR_list.extend(x)
Requirements = (pd.Series(AR_list).value_counts()/10*100).sort_values(ascending = False)[:20].to_frame()  

sns.set_style('whitegrid')

Fig_24 = plt.figure(figsize=(6,7))
Fig_24 = sns.barplot(x=Fundamentals[0], y= Fundamentals.index, alpha=0.8, 
                    palette = "terrain")
plt.title("Which courses are necessary?", fontsize=18)
plt.xlabel("Importance (%)", fontsize=16)
plt.gca().set_yticklabels(Fundamentals.index, fontsize = 14, rotation = 0)
plt.show()
Fig_25 = plt.figure(figsize=(6,7))
Fig_25 = sns.barplot(x=Requirements[0], y= Requirements.index, alpha=0.8, 
                    palette = "afmhot")
plt.title("Application Requirements", fontsize=18)
plt.xlabel("Importance (%)", fontsize=16)
plt.gca().set_yticklabels(Requirements.index, fontsize = 14, rotation = 0)
plt.show()


# In[40]:


# How much importance is given to projects?
bus_proj_imp=business_curr['Projects'].value_counts()
bus_labels = (np.array(bus_proj_imp.index))
bus_values = (np.array(bus_proj_imp.values))

tech_proj_imp=technical_curr['Projects'].value_counts()
tech_labels = (np.array(tech_proj_imp.index))
tech_values = (np.array(tech_proj_imp.values))


Fig_26 = {"data":[{"values":bus_values,"labels":bus_labels, "domain":{"x": [0,.48]},
               "name":"Managerial","hoverinfo":"label+percent","hole":0.35,"type":"pie"},
              {"values":tech_values,"labels":tech_labels, "domain":{"x": [0.52,1]},
               "name":"Technical","hoverinfo":"label+percent","hole":0.35,"type":"pie"}],
       "layout":{"title":"Do existing curriculums feature project-works?", "annotations":[
                {"font":{"size":16},"showarrow":False,"text":"Managerial","x":0.175,"y":0.5},
                {"font":{"size":16},"showarrow":False,"text":"Technical","x":0.805,"y":0.5}]
                }}
py.iplot(Fig_26, filename="Project_Imp")


# We also handpicked few top organizations and looked into their usual job positions. We evaluated their formal degree requirements as well as language requirements, which is presented below:

# In[41]:


topjobs = pd.read_csv('Top_Industry_jobs.csv', encoding="ISO-8859-1", low_memory = False)
topjobs.head()


# In[42]:


# Degree importance
deg_imp=topjobs['degree_u'].value_counts()
deg_labels = (np.array(deg_imp.index))
deg_values = (np.array(deg_imp.values))

Fig_27 = {"data":[{"values":deg_values,"labels":deg_labels, "domain":{"x": [0.25,.75]},
               "name":"Managerial","hoverinfo":"label+percent","hole":0.5,"type":"pie"}],
       "layout":{"title":"<b>Formal Degrees? Are they important?</b>", "annotations":[
                {"font":{"size":14},"showarrow":False,"text":"<b>Degree\nRequirements</b>","x":0.5,"y":0.5}]
                }}
py.iplot(Fig_27, filename="deg_Imp")


# In[43]:


#!pip install squarify
import squarify
import matplotlib
plt.style.use('fivethirtyeight')
sns.set_style('white')
langs = topjobs['Programing languages'].str.split(',')
langs_list = []
for i in langs.dropna():
    langs_list.extend(i)
programming_lang = pd.Series(langs_list).value_counts().sort_values(ascending = False)[:12].to_frame()  

Fig_28 = plt.figure(figsize=(14,8))
squarify.plot(label=programming_lang.index, sizes=programming_lang.values, color=sns.color_palette('Spectral',15))
plt.rc('font', size=18)  
plt.title("Featured programming language requirements\n", fontsize=18)
plt.axis('off')
plt.show()


# Based on these curriculum analysis and top data science job market features, the business and technical program curriculums are presented in detail in the report 

# # How EdTech Programs Are Helping?

# Beside university edcuation, Massive Online Open Courses (MOOCs) and other EdTechs are helping out emerging Data Scientists by providing a platform to learn, share, communicate, and grow. Let's have a look into those programs.

# In[44]:


# First, we looked how EdTechs helped people to motivate toward Data Science?
train_starter=mcdata['FirstTrainingSelect'].value_counts()
ts_labels = (np.array(train_starter.index))
ts_values = (np.array(train_starter.values))
trace = go.Pie(labels=ts_labels, values=ts_values,hoverinfo='label+percent',
               textfont=dict(size=20), showlegend=True,marker=dict(line=dict(color='#000000', width=0.5)))

layout = go.Layout(title='<b>How usually data scientists begin their journey</b>?', 
                   legend = dict(x = .75, y = 0.9, traceorder = 'normal', orientation = 'v'))
data_trace = [trace]
Fig_31 = go.Figure(data=data_trace, layout=layout)
py.iplot(Fig_31, filename="Learning_source")


# It seems that a significant number of professionals actually started their Data Science journey with online courses and blogs. Next, we looked closely into the MOOC and Blogs available for data scientists and presented a usage distribution.

# In[135]:


# Usage of MOOCs and Blogs from Kaggle Dataset users
MOOCcourse=mcdata['CoursePlatformSelect'].str.split(',')
course_plat=[]
for i in MOOCcourse.dropna():
    course_plat.extend(i)
course_plat=pd.Series(course_plat).value_counts()
MOOC_labels=course_plat.index
MOOC_sizes=course_plat.values

blogs=mcdata['BlogsPodcastsNewslettersSelect'].str.split(',')
blogs_fam=[]
for i in blogs.dropna():
    blogs_fam.extend(i)
blogs_fam=pd.Series(blogs_fam).value_counts()
blog_labels=blogs_fam[:5].index
blog_sizes=blogs_fam[:5].values


Fig_32 = {"data": [{"values": MOOC_sizes, "labels": MOOC_labels, "domain": {"x": [0, .48]},
          "name": "MOOCs", "hoverinfo":"label+percent+name", "hole": .5,"type": "pie"},     
         {"values": blog_sizes, "labels": blog_labels,"text":"CO2","textposition":"inside","domain": {"x": [.54, 1]},
          "name": "Blog", "hoverinfo":"label+percent+name","hole": .5, "type": "pie"}],
          "layout": {"title":"<b>MOOCs and Blogs Usage</b>", "showlegend":True, "legend":dict(x = 1.05, y = 0.8, traceorder = 'normal', orientation = 'v'),
                     "annotations": [{"font": {"size": 12},"showarrow": False,"text": "MOOC's","x": 0.19,"y": 0.5},
                                     {"font": {"size": 12}, "showarrow": False, "text": "BLOGS", "x": 0.82, "y": 0.5}]}}
py.iplot(Fig_32, filename='MOOCs and Blogs')


# # ideaLink
# ### A new proposed EdTech bridging students with industries
# 
# It seems that there are lots of help available for emerging data scientists - through University education as well as different EdTech programs such as MOOCs and Blogs. But are they sufficient enough to deal with the major challenges of Data Science? Let's look into the top challenges for a data scientist and decide ourselves.

# In[138]:


challenges = mcdata['WorkChallengesSelect'].str.split(',')
chal_list = []
for i in challenges.dropna():
    chal_list.extend(i)
challenge = pd.Series(chal_list).value_counts().sort_values(ascending = False)[:20].to_frame()  

challenge_trace = go.Bar(x = challenge.index, y = challenge[0], marker = 
                      dict(color = challenge[0], colorscale = 'Portland', 
                           showscale = False), opacity = 0.8, width = 0.6)
challenge_layout = go.Layout(title = '<b>Major Work Challenges in Data Science</b>',
                        yaxis = dict(title = 'Survey Counts', zeroline = False),
                        xaxis = dict(showticklabels = True, tickangle = 25, tickfont= dict(
                        family ='Arial, sans-serif', size = 12, color = 'black')),
                        autosize = False, width = 900, height = 700, margin=go.Margin(l=100,r=150,b=250, t=60,pad=4))
Fig_33 = go.Figure(data = [challenge_trace], layout = challenge_layout)
py.iplot(Fig_33, filename = 'challenges_major')


# It is apparent that data pre-processing is the most significant challenge of data scientists. Also, the fact that the industry needs more data scientists is quite clearly depicted. Data pre-processing as a qualification cannot be gained through educational engagement, it can be excelled from lots of practice and directly associated with professional experiences. Other than these, we see that storytelling, tool limitations, data accessibility, team working, inability to integrate findings into organization's decision-making processes, and co-ordinations are also some of the difficulties people face when working with data. Such challenges, again, can be overcome by practice, projects, conference and business scale presentations, and finally, through internship programs. That's why we are proposing a new Intern based EdTech ideaLink to support Data Scientists. The main objective of the program will be bridging Technical and Business Data Science Master's program students and recent graduates with the industry through internships. These internships will allow the students:
# - Work with professionals and learn handling critical challenges
# - Practice data based algorithms more and data analytics for real-world problems
# - Improve team work capability and large-scale data handling
# - Improve critical thinking and problem solving skills
# And eventually make the graduates of Data Science Master's programs more suiable to the job market with hands-on experiences as well as superior business and technical skills.

# In[ ]:




