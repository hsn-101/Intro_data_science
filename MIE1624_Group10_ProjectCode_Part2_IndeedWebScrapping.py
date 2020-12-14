#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install BeautifulSoup4')
import csv
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib as plt
import re
import sys
import html
import numpy as np
import nltk
nltk.download('stopwords')
from time import sleep # To prevent overwhelming the server between connections
from collections import Counter # Keep track of our term counts
from nltk.corpus import stopwords # Filter out stopwords, such as 'the', 'or', 'and'
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system('pip install plotly')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[2]:


# Function to remove stop words
import nltk
from nltk.corpus import stopwords
def remove_stop_words(text):
    # Open stop_words.txt file as a list
    #stopwords = open('stop_words.txt', 'r').read().split()       
    stopwords = set(nltk.corpus.stopwords.words('english'))
    #text = text.split()  
    #text = [w for w in text if not w in stopwords]
    #concatenate = " ".join(w for w in text if not w in stopwords)     
    #return concatenate
print(stopwords)


# In[3]:


#Function to clean individual job postings
#Input - Target URL
#Output - Cleaned text
def clean_posting(targetURL):
    try:
        target = urllib.request.urlopen(targetURL).read() # Connect to the job posting
    except: 
        return   # In case of a broken link
    
    soupysoup = BeautifulSoup(target,'html.parser') # Get the html details
   
    # Remove script and style from bs4 object
    for script in soupysoup(["script", "style"]):
        script.extract() 

    # Get text from soupysoup
    text = soupysoup.get_text() 
     
    # Split into lines
    lines = (line.strip() for line in text.splitlines())
    
    # Break multi-headlines into separate lines
    chunks = (phrase.strip() for line in lines for phrase in line.split("  ")) 
       
    def chunk_split(chunk):
        chunk_output = chunk + ' ' # Fix spacing
        return chunk_output  
    
    # Get rid of all blank lines and ends of line
    text = ''.join(chunk_split(chunk) for chunk in chunks if chunk).encode('utf-8') 
       
    #Decode the text
    try:
        text = text.decode('utf-8')
    except:                                                            
        return 
    
    # Letter format
    text = re.sub("[^a-zA-Z.+3]"," ", text)
    
    # Convert to lowercase
    text = text.lower().split()
    
    # Remove stopwords
    
    # Open stop_words.txt file as a list
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text = [w for w in text if not w in stopwords]
       
    
    # Make a list
    text = list(set(text))
    
    return text


# In[4]:


#Test the output
targetURL = 'https://www.indeed.ca/jobs?q=data%20scientist&l=Vancouver%2C%20BC&vjk=0e205ec309ab3432'
sample = clean_posting(targetURL)
sample[0:20]


# In[6]:


#Searching for the key skills on indeed.ca
def key_skills_canada():
    URL = 'https://www.indeed.ca/jobs?q=data+scientist&l='
    #URL = 'https://www.indeed.ca/jobs?q=data+scientist&l=Vancouver%2C+BC'
    # Open the search result
    html = urllib.request.urlopen(URL).read() 
    
    # Get the html from the first page
    soupy = BeautifulSoup(html,'html.parser') 
    
    #Count the number of jobs found
    count_jobs = soupy.find(id = 'searchCount').string.encode('utf-8')
    #print(count_jobs)
    num_jobs = re.findall('\d+', str(count_jobs))
    
    if len(num_jobs) > 3: # If number of jobs is greater than 1000
        total_num_jobs = (int(num_jobs[2])*1000) + int(num_jobs[3])
    else:
        total_num_jobs = int(num_jobs[2])     
    print('There were', total_num_jobs, 'jobs found.')
    
    num_pages = int(total_num_jobs/20) 
    job_descriptions = [] 
   
    # Loop through all the search result pages
    for i in range(2,int((num_pages+1))): 
        print ('Getting page', i)
        start_num = str(i*20)
        current_page = ''.join([URL, '&start=', start_num])
        #print(current_page)
        # Get the page    
        html_page = urllib.request.urlopen(current_page).read()
        # Locate all of the job links    
        soupy_page = BeautifulSoup(html_page,'html.parser')
        # The center column on the page where the job postings exist
        links = soupy_page.find(id = 'resultsCol') 
        # Get the URLS for the jobs    
        job_URLS = [str('https://www.indeed.ca') + str(link.get('href')) for link in links.find_all('a')] 
        # Only get the job related URLS    
        job_URLS = list(filter(lambda x:'clk' in x, job_URLS)) 
       
        
        for j in range(0,len(job_URLS)):
            final_description = clean_posting(job_URLS[j])
            if final_description: 
                job_descriptions.append(final_description)
            sleep(1) 
        
    print('Completed collecting job postings.')    
    print ('There were', len(job_descriptions), 'job descriptions successfully found.')
    
    # Create a counter for the search terms
    freqq = Counter()  
    [freqq.update(item) for item in job_descriptions]
    
    # Obtain the terms corresponding to particular types of skills and store them in a dictionary. 
    # Skills include technical skills and managerial skills.
    
    prog_lang_dict = Counter({'Python':freqq['python'],'R':freqq['r'],
                              'C++':freqq['c++'],'C#':freqq['c#'],
                              'Java':freqq['java'], 'PHP':freqq['php'],
                              'Ruby':freqq['ruby'], '.Net':freqq['.net'],
                              'Matlab':freqq['matlab'],'Perl':freqq['perl'], 
                              'JavaScript':freqq['javascript'], 'Scala': freqq['scala'],
                              'Swift':freqq['swift'], 'Go':freqq['go']})
                      
    analysis_dict = Counter({'Excel':freqq['excel'],'Tableau':freqq['tableau'],
                             'D3.js':freqq['d3.js'], 'SAS':freqq['sas'],
                             'SPSS':freqq['spss'], 'D3':freqq['d3'],
                             'QlikView':freqq['qlikview'],'splunk':freqq['splunk']})  
    database_dict = Counter({'SQL':freqq['sql'], 'NoSQL':freqq['nosql'],
                             'HBase':freqq['hbase'], 'Cassandra':freqq['cassandra'],
                             'MongoDB':freqq['mongodb'], 'Azure':freqq['azure'],
                            'IBM DB2':freqq['db2'], 'Access':freqq['access']})
    ML_dict = Counter({'Supervised Learning':freqq['supervised learning'], 'Time Series':freqq['time series'],
                       'Unsupervised Learning':freqq['unsupervised learning'], 'Computer Vision':freqq['computer vision'],
                       'Natural Language Processing':freqq['natural language'], 'Reinforcement Learning':freqq['reinforcement learning'],
                       'Outlier Detection':freqq['outlier'], 'Recommendation Engines':freqq['recommendation engines'], 'Pipelines':freqq['pipelines']})
    
    MLA_dict = Counter({'Logistic Regression':freqq['logistic regression'], 'Decision Trees':freqq['decision trees'],
                       'SVM':freqq['support vector'], 'Bayesian Techniques':freqq['bayesian'], 'Neural Networks':freqq['neural network'],
                       'Ensemble Methods':freqq['ensemble'], 'Gradient Boosting':freqq['gradient boosting']})
    
    hadoop_dict = Counter({'Hadoop':freqq['hadoop'], 'MapReduce':freqq['mapreduce'],
                           'Spark':freqq['spark'], 'Pig':freqq['pig'],
                           'Hive':freqq['hive'], 'Shark':freqq['shark'],
                           'Oozie':freqq['oozie'], 'ZooKeeper':freqq['zookeeper'],
                           'Flume':freqq['flume'], 'Mahout':freqq['mahout']})
                
    management_dict = Counter({'Leadership':freqq['leadership'],'Visualization':freqq['visualization'],
                               'Working with cross-functional teams':freqq['cross-functional'],
                               'Planning':freqq['planning'], 'Organizing':freqq['organizing'],
                               'Decision-making':freqq['decision-making'], 
                               'Communication':freqq['communication'],
                               'Negotiation':freqq['negotiation'], 
                               'Presentation skills':freqq['presentation'],
                               'Multitasking':freqq['multitasking'],
                               'Strategic thinking':freqq['strategic'],
                               'Analytical skills':freqq['analytical'],'Lean':freqq['lean'],
                               'Six sigma':freqq['sigma'],
                               'Continuous improvement':freqq['continuous']})
    
    project_management_dict = Counter({'PMP':freqq['pmp'], 'PMR':freqq['pmr'],
                               'PMA':freqq['pma'],
                               'Waterfall':freqq['waterfall'],
                               'Agile':freqq['agile'], 'Scrum':freqq['scrum'],
                               'Project-management':freqq['project-management']})
    
    location = Counter({'Toronto':freqq['toronto'],
                        'London':freqq['london'],
                        'Vancouver':freqq['vancouver'],
                        'Montreal':freqq['montreal'],
                        'Halifax':freqq['halifax'],
                        'Calgary':freqq['calgary'],
                        'Edmonton':freqq['edmondon']})
    
    # Combine the counter objects for skill sets          
    all_skills = prog_lang_dict + analysis_dict + hadoop_dict + database_dict + management_dict+ ML_dict + MLA_dict
    technical_skills = prog_lang_dict + analysis_dict + hadoop_dict + database_dict + ML_dict + MLA_dict
    management_skills = management_dict + project_management_dict
    
    # Create a pandas dataframe for the list of all the terms and their frequencies
    framey = pd.DataFrame(columns = ['Term', 'NumPostings'])
    framey = framey.append(all_skills, ignore_index=True)
    framey = pd.DataFrame.from_dict(all_skills, orient='index').reset_index()
    framey = framey.rename(columns={'index':'Term', 0:'NumPostings'})
    framey.NumPostings = (framey.NumPostings)*100/len(job_descriptions) 
    framey.sort_values(by = ['NumPostings'], ascending = False, inplace = True) 
    framey.to_csv('all_skills.csv',encoding = 'utf-8', index=False)
    #return framey 

    # Create a pandas dataframe for the list of technical skills and their frequencies
    framey_tech = pd.DataFrame(columns = ['tech_Term', 'tech_NumPostings'])
    framey_tech = framey_tech.append(technical_skills, ignore_index=True)
    framey_tech = pd.DataFrame.from_dict(technical_skills, orient='index').reset_index()
    framey_tech = framey_tech.rename(columns={'index':'tech_Term', 0:'tech_NumPostings'})
    framey_tech.tech_NumPostings = (framey_tech.tech_NumPostings)*100/len(job_descriptions) 
    framey_tech.sort_values(by = ['tech_NumPostings'], ascending = False, inplace = True) 
    framey_tech.to_csv('tech_skills.csv',encoding = 'utf-8', index=False)
    
    # Create a pandas dataframe for the list of management skills and their frequencies
    framey_manager = pd.DataFrame(columns = ['manage_Term', 'manage_NumPostings'])
    framey_manager = framey_manager.append(management_skills, ignore_index=True)
    framey_manager = pd.DataFrame.from_dict(management_skills, orient='index').reset_index()
    framey_manager = framey_manager.rename(columns={'index':'manage_Term', 0:'manage_NumPostings'})
    framey_manager.manage_NumPostings = (framey_manager.manage_NumPostings)*100/len(job_descriptions) 
    framey_manager.sort_values(by = ['manage_NumPostings'], ascending = False, inplace = True) 
    framey_manager.to_csv('management_skills.csv',encoding = 'utf-8', index=False)
    
    # Create a pandas dataframe for the list of cities and their frequencies
    framey_city = pd.DataFrame(columns = ['Location', 'loc_NumPostings'])
    framey_city = framey_city.append(location, ignore_index=True)
    framey_city = pd.DataFrame.from_dict(location, orient='index').reset_index()
    framey_city = framey_city.rename(columns={'index':'Location', 0:'loc_NumPostings'})
    framey_city.loc_NumPostings = (framey_city.loc_NumPostings)*100/len(job_descriptions) 
    framey_city.sort_values(by = ['loc_NumPostings'], ascending = False, inplace = True) 
    framey_city.to_csv('city_result.csv',encoding = 'utf-8', index=False)
   
    # This section will merge all the dataframes together and give output a csv file containing Canadian Indeed data
    tot_1 = pd.concat([framey, framey_tech], axis = 1)
    tot_2 = pd.concat([tot_1, framey_manager], axis = 1)
    tot_df = pd.concat([tot_2, framey_city], axis = 1) 
    tot_df.to_csv('Canada_Indeed.csv',encoding = 'utf-8', index=False)
    
    #print(framey.head())
    #print(framey_tech.head())
    #print(framey_manager.head())
    #print(framey_city.head())
    return tot_df 


# In[7]:


canada_info = key_skills_canada() 


# In[8]:


# plotting using pandas dataframe
final_plot = canada_info.plot(x = 'Term', kind = 'bar', legend = None, 
                         title = 'Percentage of Data Scientist Job Ads with All Key Skills')   
final_plot.set_ylabel('Percentage Appearing in Job Ads')
fig1 = final_plot.get_figure() 


# In[16]:


#Searching for the key skills on indeed.com
def key_skills_us():
    URL = 'https://www.indeed.com/jobs?q=data+scientist&l=United+States' #US indeed.com search no location specified
        # Open the search result
    html = urllib.request.urlopen(URL).read() 
    
    # Get the html from the first page
    soupy = BeautifulSoup(html,'html.parser') 
    
    #Count the number of jobs found
    count_jobs = soupy.find(id = 'searchCount').string.encode('utf-8')
    #print(count_jobs)
    num_jobs = re.findall('\d+', str(count_jobs))
    
    if len(num_jobs) > 3: # If number of jobs is greater than 1000
        total_num_jobs = (int(num_jobs[2])*1000) + int(num_jobs[3])
    else:
        total_num_jobs = int(num_jobs[2])     
    print('There were', total_num_jobs, 'jobs found.')
    
    num_pages = int(total_num_jobs/20) 
    job_descriptions = [] 
   
    # Loop through all the search result pages
    for i in range(2,int((num_pages+1))): 
        print ('Getting page', i)
        start_num = str(i*20)
        current_page = ''.join([URL, '&start=', start_num])
        #print(current_page)
        # Get the page    
        html_page = urllib.request.urlopen(current_page).read()
        # Locate all of the job links    
        soupy_page = BeautifulSoup(html_page,'html.parser')
        # The center column on the page where the job postings exist
        links = soupy_page.find(id = 'resultsCol') 
        # Get the URLS for the jobs    
        job_URLS = [str('https://www.indeed.com') + str(link.get('href')) for link in links.find_all('a')] 
        # Only get the job related URLS    
        job_URLS = list(filter(lambda x:'clk' in x, job_URLS)) 
       
        
        for j in range(0,len(job_URLS)):
            final_description = clean_posting(job_URLS[j])
            if final_description: 
                job_descriptions.append(final_description)
            sleep(1) 
        
    print('Completed collecting job postings.')    
    print ('There were', len(job_descriptions), 'job descriptions successfully found.')
    
    # Create a counter for the search terms
    freqq = Counter()  
    [freqq.update(item) for item in job_descriptions]
    
    # Obtain the terms corresponding to particular types of skills and store them in a dictionary. 
    # Skills include technical skills and managerial skills.
    
    prog_lang_dict = Counter({'Python':freqq['python'],'R':freqq['r'],
                              'C++':freqq['c++'],'C#':freqq['c#'],
                              'Java':freqq['java'], 'PHP':freqq['php'],
                              'Ruby':freqq['ruby'], '.Net':freqq['.net'],
                              'Matlab':freqq['matlab'],'Perl':freqq['perl'], 
                              'JavaScript':freqq['javascript'], 'Scala': freqq['scala'],
                              'Swift':freqq['swift'], 'Go':freqq['go']})
                      
    analysis_dict = Counter({'Excel':freqq['excel'],'Tableau':freqq['tableau'],
                             'D3.js':freqq['d3.js'], 'SAS':freqq['sas'],
                             'SPSS':freqq['spss'], 'D3':freqq['d3'],
                             'QlikView':freqq['qlikview'],'splunk':freqq['splunk']})  
    database_dict = Counter({'SQL':freqq['sql'], 'NoSQL':freqq['nosql'],
                             'HBase':freqq['hbase'], 'Cassandra':freqq['cassandra'],
                             'MongoDB':freqq['mongodb'], 'Azure':freqq['azure'],
                            'IBM DB2':freqq['db2'], 'Access':freqq['access']})
    ML_dict = Counter({'Supervised Learning':freqq['supervised learning'], 'Time Series':freqq['time series'],
                       'Unsupervised Learning':freqq['unsupervised learning'], 'Computer Vision':freqq['computer vision'],
                       'Natural Language Processing':freqq['natural language'], 'Reinforcement Learning':freqq['reinforcement learning'],
                       'Outlier Detection':freqq['outlier'], 'Recommendation Engines':freqq['recommendation engines'], 'Pipelines':freqq['pipelines']})
    
    MLA_dict = Counter({'Logistic Regression':freqq['logistic regression'], 'Decision Trees':freqq['decision trees'],
                       'SVM':freqq['support vector'], 'Bayesian Techniques':freqq['bayesian'], 'Neural Networks':freqq['neural network'],
                       'Ensemble Methods':freqq['ensemble'], 'Gradient Boosting':freqq['gradient boosting']})
    
    hadoop_dict = Counter({'Hadoop':freqq['hadoop'], 'MapReduce':freqq['mapreduce'],
                           'Spark':freqq['spark'], 'Pig':freqq['pig'],
                           'Hive':freqq['hive'], 'Shark':freqq['shark'],
                           'Oozie':freqq['oozie'], 'ZooKeeper':freqq['zookeeper'],
                           'Flume':freqq['flume'], 'Mahout':freqq['mahout']})
                
    management_dict = Counter({'Leadership':freqq['leadership'],'Visualization':freqq['visualization'],
                               'Working with cross-functional teams':freqq['cross-functional'],
                               'Planning':freqq['planning'], 'Organizing':freqq['organizing'],
                               'Decision-making':freqq['decision-making'], 
                               'Communication':freqq['communication'],
                               'Negotiation':freqq['negotiation'], 
                               'Presentation skills':freqq['presentation'],
                               'Multitasking':freqq['multitasking'],
                               'Strategic thinking':freqq['strategic'],
                               'Analytical skills':freqq['analytical'],'Lean':freqq['lean'],
                               'Six sigma':freqq['sigma'],
                               'Continuous improvement':freqq['continuous']})
    
    project_management_dict = Counter({'PMP':freqq['pmp'], 'PMR':freqq['pmr'],
                               'PMA':freqq['pma'],
                               'Waterfall':freqq['waterfall'],
                               'Agile':freqq['agile'], 'Scrum':freqq['scrum'],
                               'Project-management':freqq['project-management']})
    
    location = Counter({'New+York':freqq['new+york'],
                        'Chicago':freqq['chicago'],
                        'San+Francisco':freqq['san+francisco'],
                        'Cambridge':freqq['cambridge'],
                        'Miami':freqq['miami'],
                        'Seattle':freqq['seattle'],
                        'Los+Angeles':freqq['los+angeles'],
                        'Washington':freqq['washington'],
                        'Portland':freqq['portland'],
                        'Boston':freqq['boston'],
                        'Austin':freqq['austin'],
                        'Houston':freqq['houston'],
                        'Charlotte':freqq['charlotte']})
    
        # Combine the counter objects for skill sets          
    all_skills = prog_lang_dict + analysis_dict + hadoop_dict + database_dict + management_dict+ ML_dict + MLA_dict
    technical_skills = prog_lang_dict + analysis_dict + hadoop_dict + database_dict + ML_dict + MLA_dict
    management_skills = management_dict + project_management_dict
    
    # Create a pandas dataframe for the list of all the terms and their frequencies
    framey = pd.DataFrame(columns = ['Term', 'NumPostings'])
    framey = framey.append(all_skills, ignore_index=True)
    framey = pd.DataFrame.from_dict(all_skills, orient='index').reset_index()
    framey = framey.rename(columns={'index':'Term', 0:'NumPostings'})
    framey.NumPostings = (framey.NumPostings)*100/len(job_descriptions) 
    framey.sort_values(by = ['NumPostings'], ascending = False, inplace = True) 
    framey.to_csv('all_skills.csv',encoding = 'utf-8', index=False)
    #return framey 

    # Create a pandas dataframe for the list of technical skills and their frequencies
    framey_tech = pd.DataFrame(columns = ['tech_Term', 'tech_NumPostings'])
    framey_tech = framey_tech.append(technical_skills, ignore_index=True)
    framey_tech = pd.DataFrame.from_dict(technical_skills, orient='index').reset_index()
    framey_tech = framey_tech.rename(columns={'index':'tech_Term', 0:'tech_NumPostings'})
    framey_tech.tech_NumPostings = (framey_tech.tech_NumPostings)*100/len(job_descriptions) 
    framey_tech.sort_values(by = ['tech_NumPostings'], ascending = False, inplace = True) 
    framey_tech.to_csv('tech_skills.csv',encoding = 'utf-8', index=False)
    
    # Create a pandas dataframe for the list of management skills and their frequencies
    framey_manager = pd.DataFrame(columns = ['manage_Term', 'manage_NumPostings'])
    framey_manager = framey_manager.append(management_skills, ignore_index=True)
    framey_manager = pd.DataFrame.from_dict(management_skills, orient='index').reset_index()
    framey_manager = framey_manager.rename(columns={'index':'manage_Term', 0:'manage_NumPostings'})
    framey_manager.manage_NumPostings = (framey_manager.manage_NumPostings)*100/len(job_descriptions) 
    framey_manager.sort_values(by = ['manage_NumPostings'], ascending = False, inplace = True) 
    framey_manager.to_csv('management_skills.csv',encoding = 'utf-8', index=False)
    
    # Create a pandas dataframe for the list of cities and their frequencies
    framey_city = pd.DataFrame(columns = ['Location', 'loc_NumPostings'])
    framey_city = framey_city.append(location, ignore_index=True)
    framey_city = pd.DataFrame.from_dict(location, orient='index').reset_index()
    framey_city = framey_city.rename(columns={'index':'Location', 0:'loc_NumPostings'})
    framey_city.loc_NumPostings = (framey_city.loc_NumPostings)*100/len(job_descriptions) 
    framey_city.sort_values(by = ['loc_NumPostings'], ascending = False, inplace = True) 
    framey_city.to_csv('city_result.csv',encoding = 'utf-8', index=False)
   
    # This section will merge all the dataframes together and give output a csv file containing Canadian Indeed data
    tot_1 = pd.concat([framey, framey_tech], axis = 1)
    tot_2 = pd.concat([tot_1, framey_manager], axis = 1)
    tot_df = pd.concat([tot_2, framey_city], axis = 1) 
    tot_df.to_csv('USA_Indeed.csv',encoding = 'utf-8', index=False)
    
    #print(framey.head())
    #print(framey_tech.head())
    #print(framey_manager.head())
    #print(framey_city.head())
    return tot_df 


# In[17]:


usa_info = key_skills_us()


# In[18]:


# plotting using pandas dataframe
final_plot1 = usa_info.plot(x = 'Term', kind = 'bar', legend = None, 
                         title = 'Percentage of Data Scientist Job Ads with All Key Skills in USA')   
final_plot1.set_ylabel('Percentage Appearing in Job Ads')
fig2 = final_plot1.get_figure() 


# In[ ]:




