#!/usr/bin/env python
# coding: utf-8

# # LAPTOP PRICE PREDICTION
# 
# 
# Dataset URL :
# --------------
# Dataset has been downloaded from Kaggle . 
# https://www.kaggle.com/datasets/muhammetvarl/laptop-price
# 
# 
# 
# Dataset description:
# --------------------
# This dataset contains 1303 laptop models and different features that each laptop possess . for instance RAM, weight, mermory, Price, CPU, GPU and type. Many columns are Alpha-numeric and hence lot of data cleaning is required in this dataset. 
# 
# 
# 
# Descripton of each Column :
# -----------------------
# 1. company : name of the company to which laptop belongs.
# 2. Product : This shows the model name of the laptop. for example if it is from the company apple than whether it is macbook pro or macbook air.
# 3. TypeName: gives us the information weather the laptop is gaming one or a workstation ,or any other type.
# 4. Inches: size of the laptop.
# 5. ScreenResolution: gives us X and Y resolution and tells us if it has IPS pannel or not.
# 6. Cpu: Gives us the name of the Cpu company and its size.
# 7. Ram: size of the ram
# 8. Memory: what is the storage capacity of the laptop.
# 9. Gpu: Brand of the Gpu and its model.
# 10. OpSys: which operating system does that laptop have.
# 11. weight: Gives us the weight of the laptop.
# 12. Price_euros: what is the price of that laptop.
# 
# 
# 
# # 1. Data Capture and Initial Analysis
# ---------------------------------------------------------------------------
# 
# 
# 
# 
# 
# 
# 

# In[2]:


#importing some important libraries for the project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy import stats

#setting a particular style for the graphs
sns.set_style("darkgrid")
sns.despine()
plt.style.use("seaborn-darkgrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)


# In[3]:


laptops=pd.read_csv("laptops3.csv")


# In[4]:


#defining a function which will be used later in learning the relation between various features


def visual(df, col, title, symb):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5),gridspec_kw={"height_ratios": (.2, .8)})
    ax[0].set_title(title,fontsize=18)
    sns.boxplot(x=col, data=df, ax=ax[0])
    ax[0].set(yticks=[])
    sns.histplot(x=col, data=df, ax=ax[1])
    ax[1].set_xlabel(col, fontsize=16)
    plt.axvline(df[col].mean(), color='green', linewidth=2.1, label='mean=' + str(np.round(df[col].mean(),1)) + symb)
    plt.axvline(df[col].median(), color='black', linewidth=2.1, label='median='+ str(np.round(df[col].median(),1)) + symb)
    plt.axvline(df[col].mode()[0], color='red', linewidth=2.1, label='mode='+ str(df[col].mode()[0]) + symb)
    plt.legend(bbox_to_anchor=(1, 1.03), ncol=1, fontsize=17, fancybox=True, shadow=True, frameon=True)
    plt.tight_layout()
    plt.show()


# In[5]:


#printing first few rows
laptops.head()


# In[6]:


#fetching initial statistics
laptops.describe()


# # 2. Exploratory Data Analysis and Feature Engineering 

# Understanding the dataset 
# 
# Univariate  Analysis:
# -------------------------------------
# 1. Analysing each Column value count and distribution in space.
# 2. Identifying Outliers and null values if any.
# 3. converting alpha numeric columns to the numeric columns.
# 4. using regular expression method to make new features using existing ones.
# 5. Understand the data type and visualise trend.
# 6. Understand and interpret the information recorded in each column.
# 7. Understand the correlation between features.
# 8. Identify features and target variable.

# working with Ram and Weight features 

# In[7]:


#removing unnecessary column .

laptops.drop(columns=['Unnamed: 0'],inplace=True)
## remove gb and kg from Ram and weight and convert the columns to numeric datatype.
#we will add the gb and kb to the feature name
laptops['Ram'] = laptops['Ram'].str.replace("GB", "")
laptops['Weight'] = laptops['Weight'].str.replace("kg", "")
laptops['Ram'] = laptops['Ram'].astype('int32')
laptops['Weight'] = laptops['Weight'].astype('float32')

laptops.rename(columns={'Ram': 'Ram(GB)'}, inplace=True)
laptops.rename(columns={'Weight': 'Weight(kg)'}, inplace=True)


# COLUMN :ScreenResolution 

# In[8]:


#working with screen resolution feature 
"""if we see the ScreenResolution feature there are very few laptops that are touchscreen .
we can extract the touchscreen laptops as this fcator can affect the price point .other than that the screen resolution
column gives us the information about the pixel counts which we can extract using regular expression , as other information
may end up being unused in our model ."""
laptops['Touchscreen'] = laptops['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

#extract IPS column
laptops['Ips'] = laptops['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

def findXresolution(s):
  return s.split()[-1].split("x")[0]
def findYresolution(s):
  return s.split()[-1].split("x")[1]
#finding the x_res and y_res from screen resolution
laptops['X_res'] = laptops['ScreenResolution'].apply(lambda x: findXresolution(x))
laptops['Y_res'] = laptops['ScreenResolution'].apply(lambda y: findYresolution(y))
#convert to numeric
laptops['X_res'] = laptops['X_res'].astype('int')
laptops['Y_res'] = laptops['Y_res'].astype('int')


laptops['ppi'] = (((laptops['X_res']**2) + (laptops['Y_res']**2))**0.5/laptops['Inches']).astype('float')
laptops.corr()['Price_euros'].sort_values(ascending=False)


# In[9]:


#extracting all the alphabetical information
laptops['screentype'] = laptops['ScreenResolution'].replace(r'(\d+x\d+)','',regex=True)
#removing all the extra information
laptops['screentype'] = laptops['screentype'].replace(r'(Full HD|Quad HD|Quad HD|\+|/|4K Ultra HD)','',regex=True)
laptops['screentype']


# extracting wheather a particular laptop is touchscreen or not 
# 

# In[10]:


laptops['Touchscreen'] = laptops['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[11]:


laptops['screentype'].value_counts()


# their are only two types of screen IPS panel and IPS Panel Retina, and hence we should clean the data ahead.

# In[12]:


laptops['screentype']=laptops['screentype'].replace(r' ','',regex=True)
laptops['screentype'].value_counts()


# In[13]:


#replacing the nan values
laptops['screentype'] = laptops['screentype'].replace(r'^\s*$', np.nan, regex=True)
laptops['screentype'].value_counts()


# In[14]:


#dropping the main column
laptosp= laptops.drop('ScreenResolution', axis=1)
laptops.head()


# In[15]:


#dropping all the unncessary column
laptops.drop(columns = ['ScreenResolution','X_res','Y_res'], inplace=True)
laptops.head()


# COLUMN : CPU 

# In[16]:


laptops['Cpu']


# In[17]:


"""we can get Cpu brand and its frequency in two different column in order to simplify the data
working with cpu column , this column has more unique features and that is why we will reduce the unique values of this feature 
first we will extract Name of CPU which is first 3 words from Cpu column and then we will check which processor it is"""
 # expeeriment that failed as it was not needed to sort the value of Cpu_size and the code was not readable

#sns.scatterplot(x=laptops['Cpu_size'],y=laptops['Price_euros'])
#laptops['Cpu_grade'] = [sorted(set([float
#(re.findall(r'\d*\.?\d+', a)[0]) for a in laptops['Cpu'].str
#.extract('(\d*\.?\d+GHz)')[0]])).index(s) for s in [float(re.findall(r'\d*\.?\d+', a)[0]) 
#for a in laptops['Cpu'].str.extract('(\d*\.?\d+GHz)')[0]]]


laptops['Cpu_size']=laptops['Cpu'].str.extract('(\d*\.?\d+GHz)',expand=True)
laptops['Cpu_size']=laptops['Cpu_size'].str.extract('(\d*\.?\d+)',expand=True)
laptops.info()
laptops["Cpu_size"]=laptops["Cpu_size"].astype(float)


# In[18]:


#extrating other data
laptops['Cpu']= laptops['Cpu'].str.replace(r'(\d+(?:\.\d+)?GHz)', '', regex=True)


# In[19]:


#extrating the brand name of the Cpu in that column itself.
laptops['Cpu_brand'] = laptops['Cpu'].str.extract(r'^(\w+)')
laptops['Cpu_brand'].value_counts()


# Majority of the CPu are from intel and very few are from AMD. Their is only one Cpu from samsung and we can analyse it further.

# In[20]:


laptops[laptops['Cpu_brand']=='Samsung']


# 
# Their is only one laptop from samsung brand and it may disturb our analysis and hence it will be better to drop this row.

# In[21]:


laptops=laptops.drop(1191)


# In[22]:


laptops.head()


# COLUMN : Memory

# In[23]:


laptops['Memory']


# We can extract a lot of information from this column. laptop can have one or two hard drive with same or differnt space and type. Their is SSD , HHD, Flash Storage and hybrid . we can extract all tha different column.

# In[24]:


laptops['Memory'].value_counts()


# In[25]:


#their is variety of things in the memory column , therefore we will make 4 categories like we did for Cpu brand .

laptops['Memory']=laptops['Memory'].astype(str).replace('\.0','',regex=True)
laptops['Memory']=laptops['Memory'].str.replace('GB','')
laptops['Memory']=laptops['Memory'].str.replace('TB','000')
variable = laptops['Memory'].str.split("+",n=1,expand=True)
laptops['first']=variable[0].str.strip()
#strip(): returns a new string after removing any leading and trailing whitespaces including

laptops['second']=variable[1]
laptops['Layer1HDD'] = laptops['first'].apply(lambda x:1 if 'HDD' in x else 0)
laptops['Layer1SSD'] = laptops['first'].apply(lambda x:1 if 'SSD' in x else 0)
laptops['Layer1Hybrid'] = laptops['first'].apply(lambda x:1 if 'Hybrid' in x else 0)
laptops['Layer1Flash_Storage'] = laptops['first'].apply(lambda x:1 if 'Flash Storage' in x else
0)
laptops['first']=laptops['first'].str.replace(r'\D','')
# converted 256 SSD to 256
laptops['second'].fillna('0',inplace=True)
#The fillna() function is used to fill NA/NaN values using the specified method
laptops['Layer2HDD'] = laptops['second'].apply(lambda x:1 if 'HDD' in x else 0)
laptops['Layer2SSD'] = laptops['second'].apply(lambda x:1 if 'SSD' in x else 0)
laptops['Layer2Hybrid'] = laptops['second'].apply(lambda x:1 if 'Hybrid' in x else 0)
laptops['Layer2Flash_Storage'] = laptops['second'].apply(lambda x:1 if 'Flash Storage' in x else 0)
laptops['second']=laptops['second'].str.replace(r'\D','')
laptops['first']=laptops['first'].astype(int)
laptops['second']=laptops['second'].astype(int)
#here first snd second contain a number like 128 or 256 and layer1 and layer 2 contain absence or presence of HDD or SSD and
#multiplying and adding them will give the total HHD
laptops['HDD']=(laptops['first']*laptops['Layer1HDD']+laptops['second']*laptops['Layer2HDD'])
laptops['SSD']=(laptops['first']*laptops['Layer1SSD']+laptops['second']*laptops['Layer2SSD'])
laptops['Hybrid']=(laptops['first']*laptops['Layer1Hybrid']+laptops['second']*laptops['Layer2Hybrid'])
laptops['Flash_Storage']=(laptops['first']*laptops['Layer1Flash_Storage']+laptops['second']*laptops['Layer2Flash_Storage'])
#dropping all the unnecessary columns
laptops.drop(columns=['first','second','Layer1HDD','Layer1HDD','Layer2HDD','Layer1SSD','Layer2SSD','Layer1Hybrid','Layer2Hybrid','Layer2Flash_Storage','Layer1Flash_Storage'],inplace=True)

laptops.head()


# In[26]:


laptops.drop(columns=['Hybrid','Flash_Storage','Memory'],inplace=True)


# COLUMN: Gpu

# In[27]:


#extracting the GPu brand name 
laptops['Gpu_brand'] = laptops['Gpu'].str.extract(r'^(\w+)')
laptops['Gpu_brand'].value_counts()


# Majority of the Gpu are from intel or Nvidia .Very few models are from AMD. Their is only one laptop from ARM and hence we should analyze it further.

# In[28]:


laptops[laptops['Gpu_brand']=='ARM']


# In[29]:


#it is the same samsung laptop which we saw in the Cpu brand as well . it will be better if we drop this row from the dataset


# In[30]:


#laptops=laptops.drop(1191)


# In[31]:


laptops.head()


# In[32]:


laptops.info()


# In[33]:


#our data is clean  and can be saved in a different csv file
laptops.to_csv('laptop-clean.csv', index=False)


# we can clean the data a little more .
# 

# In[34]:


#creating a copy of the dataset
laptops1=pd.read_csv('laptop-clean.csv')
laptops1.head()
laptops_clean=laptops1.copy()


# In[35]:


laptops1.info()


# # 3. Research Questions

# ## 3.1 What is being Analysed?

# 1. The aim of this research is to predict the price of a laptop model based on its features like RAM, GPU, CPU, TypeName and so on.
# 2. This is a supervised machine learning algorithm as we know the features and its importance in prediction .
# 3. The dataset can also be analysed to give correlation between different features of laptops. for instance which company uses which CPU or GPU and so forth.
# 
# 
# 
# -----
# Supervised machine learning is mainly used for Regression purpose.

# ## 3.2  Why is it being Analysed ?

# Analysing Prices of the laptop based on its features can help us in following ways :
# 
# 1. Give an individual the approximate price of the laptop based on his requirement of features.
# 2. Sometimes we tend to buy laptops with higher price just because it belongs to a certain company and this model can show us that which all laptops can we buy in the same price but better features from different company.
# 3. The main idea is to find more economical option for a person keeping in mind the basic requirement of a person.

# ## 3.3 What is the future scope of this research ?

# Future scope:
# 
# 1. In future one can try to extract more information from alpha numeric columns like CPU, Memory and GPU .
# 2. Better feature Engineering can be done by making more columns such as PPI in this one , which has strong correlation with Price.
# 3. different machine learning algorithm can be used and experimented.

# ## 3.4 How is it being analysed?

# As this is a Regrression problem different regression algoritms were implemented :
# 1. Random Forest Classifier
# 2. Decision Tree
# 3. Linear Regression
# 4. XGBoost
# 5. XGBoost with Optuna Optimisation
# 6. Support Vector Regression

# ## 3.5 Identifying targets and variables for regression analysis:

# Target variable : The Price_euros feature is the target as we want to find the price of the laptops.
# 
# Labels
# ------
# We consider all the other variables to perform regression analysis. 

# # 4. Interpretting Data Through Visuals

# Analyzing Company Column

# In[36]:


fig, ax  = plt.subplots(figsize=(12,7))
ax=sns.countplot(x='Company', data=laptops, palette='mako_r', order = laptops['Company'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=80)
ax.bar_label(ax.containers[0])
plt.title('Number of laptops by brands')
plt.show()


# most sold laptops are dell , lenovo, HP and followed by Asus, Acer, MSI, Toshiba and apple. other laptops are sold in a very less quantity 

# what type of laptops is mostly used ?

# In[37]:


#checking the feautre typename and its relation with our target variable 
laptops['TypeName'].value_counts().plot(kind='bar')
plt.xticks(rotation='vertical')
plt.show()
#we will once again see the average price of the laptop for each type of laptop

sns.barplot(x=laptops['TypeName'] , y= laptops['Price_euros'])
plt.xticks(rotation='vertical')
plt.show()
"""The most selling laptops are Notebook and if we see the plot with price_euros than we understand that it is budget 
friendly laptop and hence most selling one .there are six type of laptops , the most sold laptops are notebook and
workstation, whereas netbook are least sold .

workstation are the most expensive , followed by ultrabook.
netbook and notebook are more economical"""


print('Most laptops are notebooks, which make {:.2f}% of the total laptops'.format(len(laptops[laptops['TypeName']=='Notebook'])*100/len(laptops)))


# Analyzing the size feature of the laptop

# In[38]:


#which size is most popular?
fig, ax  = plt.subplots(figsize=(10,5))
ax=sns.countplot(x='Inches', data=laptops, palette='viridis_r')
ax.set_xticklabels(ax.get_xticklabels(), rotation=80);
ax.bar_label(ax.containers[0])
plt.title('Laptop screen size (inches)')
plt.show()


# In[39]:


print('Most laptops have 15.6 inches, which make {:.2f}% of the total laptops'.format(len(laptops[laptops['Inches']==15.6])*100/len(laptops)))


# Their are laptops with very unconventional size and hence we will only keep those laptops in dataset for which the sizes are common

# In[40]:


inches_list = laptops['Inches'].value_counts().index[:6].tolist()
inches_list


# We will keep laptops with these 6 sizes!

# In[41]:


laptops_clean = laptops_clean[laptops_clean['Inches'].isin(inches_list)]


# In[42]:


fig, ax  = plt.subplots(figsize=(6,5))
ax=sns.countplot(x='Inches', data=laptops_clean, palette='viridis_r')
ax.set_xticklabels(ax.get_xticklabels(), rotation=80);
ax.bar_label(ax.containers[0])
plt.title('Laptop screen size (inches)')
plt.show()


# In[43]:


print('We removed {} outliers!'.format(len(laptops)-len(laptops_clean)))


# how is weight distributed among the laptops

# In[44]:


visual(laptops_clean, 'Weight(kg)', 'Weight Distribution','kg')


# In[45]:


#seeing the distribution of the price
visual(laptops_clean, 'Price_euros', 'Price Distribution','$')


# In[46]:


fig, ax  = plt.subplots(figsize=(5,3))
ax=sns.boxplot(x='Price_euros', data=laptops)


# Their are many outliars in the price columns and hence this may affect our analysis

# In[47]:


sns.boxplot(x='TypeName', y='Price_euros', data=laptops[laptops['Price_euros']>3000], hue='Cpu_brand')


# Most of the intel CPu are gaming and workstation. their is only one model for notebook and ultrabook with the Price of 4900 and 3200. Their are two outliars in the gaming laptops as well with the price of 5500 and 6100.

# In[48]:


laptops[laptops['Price_euros']>4500]


# The expensive laptops are razer blade pro gaming and lenovo thinkpad P51. 

# Distribution of ram among the price 

# In[49]:


visual(laptops_clean, 'Ram(GB)','RAM distribution','GB')


# In[50]:


ram_sales = laptops_clean.groupby('Ram(GB)').count().sort_values(by = 'Company', ascending = True)
sales_pct = list(map(lambda x: (x/1302)*100, ram_sales.Company ))

fig4, ax4 = plt.subplots(figsize=(10,8))
ax4.pie(sales_pct, startangle= 90, labels = list(map(lambda x: f'{x:.1f}%',sales_pct)), rotatelabels= True)
ax4.legend(list(map(lambda x: f'{x} GB RAM',ram_sales.index)), loc = 'best')
plt.show()


# In[51]:



print('Most laptops have 8 GB RAM, which make {:.2f}% of the total laptops'.format(len(laptops[laptops['Ram(GB)']==8])*100/len(laptops)))


# How is Cpu frequency distributed among people ?

# In[52]:


visual(laptops_clean, 'Cpu_size','CPU freq distribution','GHz')


# In[53]:


print('Most laptops have 2.5 GHz CPU, which make {}% of the total laptops'.format(np.round(len(laptops[laptops['Cpu_size']==2.5])*100/len(laptops),2)))


# # 5. Multivariate Outliars Detection

#  Price vs RAM

# In[54]:


#we define our own function to get the linear relation between two variables and plot it.

def lr_plot(df, col_x, col_y, leg):
    slope, intercept, r_value, p_value, std_err = stats.linregress(laptops[col_x],laptops[col_y])
    sns.regplot(x=col_x, y = col_y, data=laptops, color='#0d98ba', line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})
    plt.legend(loc=leg, ncol=1, fontsize=15, fancybox=True, shadow=True, frameon=True)
    plt.title(col_y + ' VS ' + col_x)
    plt.show()

    return slope, intercept


# In[55]:


slope, intercept = lr_plot(laptops_clean,'Ram(GB)','Price_euros', 'lower right')


# In[56]:


laptops_clean[laptops_clean['Ram(GB)']>60]


# The outlier value is a very high end gaming ASUS pc. We can drop it.

# In[57]:


laptops_clean = laptops_clean[laptops_clean['Ram(GB)']<60]


# In[58]:


slope, intercept = lr_plot(laptops_clean,'Ram(GB)','Price_euros', 'upper left')


# By removing the outlier value we can see that the slope increases and the intercept decreases.
# 
# According to the slope, every GB of RAM added on the PC adds roughly 107$ to the laptop value.

# Price VS CPU Brand VS GPU Brand

# In[59]:


cpu_palette = {'Intel':'#0d98ba', 'AMD':'#FF0000', 'Nvidia':'#46C646'}
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x='Cpu_brand', y='Price_euros', data=laptops, hue='Gpu_brand', palette=cpu_palette)
ax.set_title('Price vs CPU brand by GPU brand')
plt.ylabel('price ($)')
plt.legend(loc='upper right', ncol=1, fontsize=15, fancybox=True, shadow=True, frameon=True)
plt.title('Price VS CPU brand by GPU brand')
plt.show()


# Insights from this plot:
# 
# 1. Laptops with Intel CPUs are more expensive.
# 2. Laptops with an AMD CPUs also have and AMD GPUs are generally more economical.
# 3. Laptops with Nvidia GPUs are more expensive , which we saw when we were dealing with GPU column.
# 

# Which are the TOP 15 most common CPUs?
# 

# In[60]:


cpu_list = laptops_clean['Cpu'].value_counts()[:15].index.tolist()


# In[61]:


plt.figure(figsize=(8,6))
ax=sns.countplot(x='Cpu', data=laptops_clean[laptops_clean['Cpu'].isin(cpu_list)], order = cpu_list, palette='viridis')
plt.xticks(rotation=80);
ax.bar_label(ax.containers[0])
plt.title('TOP 15 common CPUs')
plt.xlabel('')
plt.show()


# we can see that all the top 15 CPU are from intel.

# What is the average price of laptops by company?
# 

# In[62]:


laptops['Company'].value_counts()


# considering companies with more number of laptops

# In[63]:


company_list = laptops['Company'].value_counts().index[:8].tolist()
company_list


# In[64]:


plt.figure(figsize=(9,5))
ax=sns.barplot(x='Company', y='Price_euros', data=laptops_clean[laptops_clean['Company'].isin(company_list)],
                order=company_list, 
                palette='Spectral', 
                ci=False,
                edgecolor="black")
plt.xticks(rotation=80);
ax.bar_label(ax.containers[0])
plt.title('Average price of laptops by company')
plt.show()


# 1. MSI laptops are more expensive than others followed by apple and toshiba.
# 2. Acer laptops are more economical with the price of 626.776

# ## Correlation Marix

# In[65]:


plt.figure(figsize=(6,5))
sns.heatmap(laptops_clean.corr(), cmap='RdBu', annot=True, vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


# Insights from correlation matrix
# 1. inches and weight are highly correlated ,which is very obvious that a bigger laptop will have more weight.
# 2. Ram and price euros have a good correlation , laptops with higher ram are more expensive
# 3. Price is fairly correlated with SSD , that signifies that having a SSD drive can increase the price of the laptop.
# 

# # Data Preparation For ML Modeling

# We will use clean file for data pre-processing

# In[66]:


laptops1=laptops_clean.copy()


# In[67]:


sns.displot(laptops1['Price_euros'])


# In[68]:


#we need to make the data symmetric as it is skewed

laptops1['Price_euros']=np.log(laptops1['Price_euros'])


# In[69]:


sns.displot(laptops1['Price_euros'])


# In[70]:


laptops1=laptops1.fillna('NaN')


# # 6. Categorical feature encoding

# In[71]:


laptops1.info()


# Their are total 10 variables with object datatype .To deal with this variable we can convert this to numerical data type by one hot encoding as they are not ordinal data type. But performng One hot enchoding to this many features will lead to so many extra columns and rows which may affect our analysis and therefore we will consider label encoding too.

# In[72]:


catCols =  ['Company','Product','TypeName','Cpu','Gpu','OpSys','screentype','Gpu_brand','Cpu_brand']


# In[73]:


#One hot encoding
print('Dataframe encoded by OHE dimension : ', pd.get_dummies(laptops1, columns=catCols, drop_first=True).shape)


# In[74]:


#Label encoding
en = LabelEncoder()
for cols in catCols:
    laptops1[cols] = en.fit_transform(laptops1[cols])
print('Dataframe encoded by Label encoding dimension : ', laptops1.shape)


# In[75]:


laptops1.head()


# # 7. Data Preparation

# In[76]:


X=laptops1.drop('Price_euros', axis = 1).values


# In[77]:


y=laptops1['Price_euros'].values


# ##  Train -Test split

# importing important libraries

# In[78]:


import regex as re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

import xgboost
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
#!pip install optuna 
import optuna

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

seed=42


# In[79]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed)


# Train Validation Split

# In[80]:


#Moreover, we define an additional validation set, which will be used to monitor overfitting.

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = seed)


# # 8. Training Models
# 
# ## 8.1  Random Forest Regressor

# In[81]:


rf = RandomForestRegressor(n_estimators=100, max_depth=100, max_features=15)


# In[82]:


rf.fit(X_train,y_train)


# In[83]:


y_pred_rf = rf.predict(X_test)


# In[84]:


mse_rf = mean_squared_error(np.exp(y_test), np.exp(y_pred_rf))
print("RMSE using RF: {} $ ".format(np.round(np.sqrt(mse_rf)),4))


# In[85]:


print("R2 using Random Forest: {:.2f} %".format(np.round(r2_score(y_test, y_pred_rf),4)*100))


# ## 8.2 Decision Tree Regressor

# In[86]:


dt = DecisionTreeRegressor( max_depth=500, max_features=15)


# In[87]:


dt.fit(X_train,y_train)


# In[88]:


y_pred_dt = dt.predict(X_test)


# In[89]:


mse_dt = mean_squared_error(np.exp(y_test), np.exp(y_pred_dt))
print("RMSE using Decision Tree: {} $ ".format(np.round(np.sqrt(mse_dt)),4))


# In[90]:


print("R2 using Decision Tree: {:.2f} %".format(np.round(r2_score(y_test, y_pred_dt),4)*100))


# ## 8.3 Linear Regression

# In[91]:


reg = LinearRegression()


# In[92]:


reg.fit(X_train,y_train)


# In[93]:


y_pred_reg = reg.predict(X_test)


# In[94]:


mse_reg = mean_squared_error(np.exp(y_test), np.exp(y_pred_reg))
print("RMSE using Linear Regression: {} $ ".format(np.round(np.sqrt(mse_reg)),4))


# In[95]:


print("R2 using Linear Regression: {:.2f} %".format(np.round(r2_score(y_test, y_pred_reg),4)*100))


# ## 8.4 XG Boost Regressor

# In[96]:


XG = XGBRegressor()


# In[97]:


XG.fit(X_train,y_train)


# In[98]:


y_pred_XG = XG.predict(X_test)
mse_XG = mean_squared_error(np.exp(y_test), np.exp(y_pred_XG))
print("RMSE using XG Boost: {} $ ".format(np.round(np.sqrt(mse_XG)),4))
print("R2 using XG Boost: {:.2f} %".format(np.round(r2_score(y_test, y_pred_XG),4)*100))


# ###  Feature Importance

# when we are dealing with random forest regressor it is better to see feature importance . It helps us to understand due to which feature our model is learning the most .
# 
# we will see those features(column name) from the x_train

# In[99]:


feature_name_list=laptops1.drop('Price_euros', axis = 1).columns


# We use the column names as the feature names, so that in the following plot we will be able to see the actual feature names

# In[100]:


rf.feature_names = feature_name_list


# In[101]:


plt.barh(rf.feature_names,rf.feature_importances_)
plt.xticks(rotation=90);
plt.title('Feature Importance by Random Forest')
plt.xlabel('Feature Importance (%)')
plt.show()


# In[102]:


dt.feature_names = feature_name_list
plt.barh(dt.feature_names,dt.feature_importances_)
plt.xticks(rotation=90);
plt.title('Feature Importance by Decision Tree')
plt.xlabel('Feature Importance (%)')
plt.show()


# Linear Regrression do not have Feature_importance as an attribute

# In[103]:


XG.feature_names = feature_name_list
plt.barh(XG.feature_names,XG.feature_importances_)
plt.xticks(rotation=90);
plt.title('Feature Importance by XGBoost')
plt.xlabel('Feature Importance (%)')
plt.show()


# ##  8.5  XGBoost Optimization with OPTUNA

# XG boost Optimisation with optuna is used as it has more accuracy in the model than XG Boost and hence gives us more precise prediction .

# In[104]:


#As optuna is a black-box optimizer that means it needs a objective function which returns a numerical value that helps in hypertunning of parameters
#trial : SPecifies which hyperparameter should be tuned it gives accuracy , which shows performance of the trial





def objective(trial, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val):
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dvalid = xgboost.DMatrix(X_val, label=y_val)

    param = {
        'objective' : 'reg:squarederror',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 5.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 5.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.001,0.01,0.05,0.1,0.2,0.25,0.3]),
        'n_estimators': trial.suggest_categorical('n_estimators', [300,400,500,1000,1500,2000,2500,3000]),
        'max_depth': trial.suggest_categorical('max_depth', [3,4,5,6,7]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
    }


    model = xgboost.XGBRegressor(**param)    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)   
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)

    return rmse


# In[105]:


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")

params = []

for key, value in trial.params.items():
    params.append(value)
    print("    {}: {}".format(key, value))


# In[106]:


params


# In[107]:


lambda_opt = params[0]
alpha_opt = params[1]
colsample_bytree_opt = params[2]
subsample_opt = params[3]
learning_rate_opt = params[4]
n_estimators_opt = params[5]
max_depth_opt = params[6]
min_child_weight_opt = params[7]


# In[108]:


xgb = XGBRegressor(reg_lambda = lambda_opt,
                   alpha = alpha_opt,
                   colsample_bytree = colsample_bytree_opt,
                   subsample_opt = subsample_opt,
                   learning_rate = learning_rate_opt,
                   n_estimators = n_estimators_opt,
                   max_depth = max_depth_opt,
                   min_child_weight = min_child_weight_opt)


# In[109]:


xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)


# In[110]:


y_pred_xgb = xgb.predict(X_test) 


# In[111]:


mse_xgb = mean_squared_error(np.exp(y_test), np.exp(y_pred_xgb))


# In[112]:


print("RMSE with XGBoost with optuna: {:.2f} $".format(np.round(np.sqrt(mse_xgb),2))) 


# In[113]:


print("R2 with XGBoost with Optuna: {:.2f} % ".format(np.round(r2_score(y_test, y_pred_xgb),4)*100))


# In[114]:


xgb.feature_names = feature_name_list


# In[115]:



plt.barh(xgb.feature_names,xgb.feature_importances_)
plt.xticks(rotation=90);
plt.title('Feature Importance by XGBoost + Optuna')
plt.xlabel('Feature Importance (%)')
plt.show()


# ## 8.6 Support Vector Regression

# In[116]:


from sklearn.svm import SVR
regressor= SVR(kernel='rbf')
regressor.fit(X_train,y_train)


# In[117]:


y_pred_regressor = regressor.predict(X_test)


# In[118]:


mse_regressor = mean_squared_error(np.exp(y_test), np.exp(y_pred_regressor))
print("RMSE using Support Vector Regression: {} $ ".format(np.round(np.sqrt(mse_regressor)),4))


# In[119]:


print("R2 using Support Vector Regression: {:.2f} %".format(np.round(r2_score(y_test, y_pred_regressor),4)*100))


# In[120]:


# since the R2 score is really low we wont consider this model


# # 9. Result summary

# In[121]:


plt.figure(figsize = (10,10))
plt.scatter(np.exp(y_test), np.exp(y_pred_rf), alpha=0.1, color='red',label='RF, R2 {:.2f} %'.format(r2_score(y_test, y_pred_rf)*100))
plt.scatter(np.exp(y_test), np.exp(y_pred_dt), alpha=0.1, color='green',label='Decision Tree, R2 {:.2f} %'.format(r2_score(y_test, y_pred_dt)*100))
plt.scatter(np.exp(y_test), np.exp(y_pred_reg), alpha=0.1, color='yellow',label='Linear Regression, R2 {:.2f} %'.format(r2_score(y_test, y_pred_reg)*100))
plt.scatter(np.exp(y_test), np.exp(y_pred_XG), alpha=0.1, color='blue',label='XGBoost, R2 {:.2f} %'.format(r2_score(y_test, y_pred_XG)*100))
plt.scatter(np.exp(y_test), np.exp(y_pred_xgb), alpha=0.1, color='black',label='XGBoost + Optuna, R2 {:.2f} %'.format(r2_score(y_test, y_pred_xgb)*100))
#plt.scatter(np.exp(y_test), np.exp(y_pred_regressor), alpha=0.1, color='purple',label='Support Vector Regressor, R2 {:.2f} %'.format(r2_score(y_test, y_pred_regressor)*100))
plt.plot([0, 7000], [0, 7000], linestyle='--')
plt.axis([0, 7000, 0, 7000])
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.title('Comparing All price regression ($)')
plt.legend(loc='upper left', ncol=1, fontsize=13, fancybox=True, shadow=True, frameon=True)
plt.show()


# Plot for Highest R2 Score

# In[122]:


plt.figure(figsize = (10,10))
plt.scatter(np.exp(y_test), np.exp(y_pred_xgb), alpha=0.5, color='black',label='XGBoost + Optuna, R2 {:.2f} %'.format(r2_score(y_test, y_pred_xgb)*100))

plt.plot([0, 7000], [0, 7000], linestyle='--')
plt.axis([0, 7000, 0, 7000])
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.title('XGBoost + Optuna price regression ($)')
plt.legend(loc='upper left', ncol=1, fontsize=13, fancybox=True, shadow=True, frameon=True)
plt.show()


# In[123]:


print("RMSE using RF: {:.2f} $ ".format(np.sqrt(mse_rf)))

print("RMSE using Decision Tree: {} $ ".format(np.round(np.sqrt(mse_dt)),4))
print("RMSE using Linear Regression: {} $ ".format(np.round(np.sqrt(mse_reg)),4))
print("RMSE using XG Boost: {} $ ".format(np.round(np.sqrt(mse_XG)),4))
print("RMSE with XGBoost : {:.2f} $".format(np.sqrt(mse_xgb)))
print("RMSE using Support Vector Regression: {} $ ".format(np.round(np.sqrt(mse_regressor)),4))


# In[128]:


print("R2 using Random Forest: {:.2f} %".format(np.round(r2_score(y_test, y_pred_rf),4)*100))
print("R2 using Decision Tree: {:.2f} %".format(np.round(r2_score(y_test, y_pred_dt),4)*100))
print("R2 using Linear Regression: {:.2f} %".format(np.round(r2_score(y_test, y_pred_reg),4)*100))
print("R2 using XG Boost: {:.2f} %".format(np.round(r2_score(y_test, y_pred_XG),4)*100))
print("R2 with XGBoost with Optuna: {:.2f} % ".format(np.round(r2_score(y_test, y_pred_xgb),4)*100))
print("R2 using Support Vector Regression: {:.2f} %".format(np.round(r2_score(y_test, y_pred_regressor),4)*100))


# In[124]:


plt.figure(figsize=(5,5))
plt.barh(rf.feature_names,rf.feature_importances_, alpha=0.3, label='RF', color='red')

plt.barh(dt.feature_names,dt.feature_importances_, alpha=0.3, label='DT', color='green')

plt.barh(XG.feature_names,XG.feature_importances_, alpha=0.3, label='XG Boost', color='blue')
plt.barh(xgb.feature_names,xgb.feature_importances_, alpha=0.3, label='XGBoost +optuna', color='black')
plt.legend(loc='center right',ncol=1, fontsize=14, fancybox=True, shadow=True, frameon=True)
plt.title('Feature Importance for price prediction\nby Different Models')
plt.xlabel('Feature Importance (%)')
plt.show()


# We can see that RF chose RAM, CPU, weight and product(name) as the most important features.
# In the case of XGBoost, the most important features are memory_1_type, RAM, resolution, cpu, typename and gpu brand.
# It is also worth noticing XGBoost didnt give as much importance to RAM and CPU as RF did. It spreaded more the feature importance among other features.

# In[125]:


xgb.feature_names = laptops1.drop('Price_euros', axis = 1).columns
feat_df= pd.DataFrame({'feature': xgb.feature_names,'importance':xgb.feature_importances_})


# In[126]:


sorted_df=feat_df.sort_values('importance', ascending=False)


# In[127]:


plt.figure(figsize=(9,5))
sns.barplot(x='importance', y='feature', data=sorted_df, palette='mako')
plt.title('Feature Importance to predict price by XGBoost + optuna')
plt.xlabel('Feature Importance (%)')
plt.ylabel('')
plt.show()

