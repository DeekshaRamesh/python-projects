
# coding: utf-8

# In[266]:

# Titanic Dataset available in Kaggle


# In[267]:

import pandas as pd
from pandas import Series, DataFrame
import os


# In[268]:

os.chdir('C:\\Users\\sri\\Desktop\\learning\\python')


# In[396]:

os.getcwd()


# In[270]:

titanic_df = pd.read_csv('train.csv')


# In[271]:

titanic_df.head()


# In[272]:

titanic_df.info()


# Some basic questions:
#     1. Who were the passengers on the ship? (Age, Gender, Class)
#     2. What desck were the passengers on and how is it related to the class? 
#     3. Where did the passengers come from?
#     4. Who was alone and who was with family?
#     5. What factors helped someone survive the sinking?
#     

# In[273]:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[274]:

sns.set_style("white")


# In[275]:

# 1. Who were the passengers on the ship?
sns.factorplot('Sex',data=titanic_df, kind="count", hue='Pclass')


# In[276]:

sns.factorplot('Pclass',data=titanic_df, kind="count", hue='Sex')


# In[277]:

def male_female_child(passenger):
    age,sex = passenger
    
    if age<16:
        return 'child'
    
    else:
        return sex


# In[278]:

titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child, axis=1)


# In[279]:

titanic_df.head(10)


# In[280]:

sns.factorplot('Pclass',data=titanic_df, kind="count", hue='person', hue_order=['child','female','male'] )


# In[281]:

titanic_df.Age.head()


# In[282]:

titanic_df['Age'].hist(bins=70)


# In[283]:

titanic_df['Age'].mean()


# In[284]:

titanic_df['person'].value_counts()


# In[285]:

import warnings 


# In[286]:

warnings.filterwarnings('ignore')


# In[287]:

fig = sns.FacetGrid(titanic_df, hue='Sex', aspect=4)

fig.map(sns.kdeplot,'Age', shade=True)

fig.set(xlim=(0,titanic_df['Age'].max()))

fig.add_legend()


# In[288]:

fig = sns.FacetGrid(titanic_df, hue='person', aspect=4)

fig.map(sns.kdeplot,'Age', shade=True)

fig.set(xlim=(0,titanic_df['Age'].max()))

fig.add_legend()


# In[289]:

fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4)

fig.map(sns.kdeplot,'Age', shade=True)

fig.set(xlim=(0,titanic_df['Age'].max()))

fig.add_legend()


# In[290]:

# Question 2


# In[291]:

titanic_df.head()


# In[292]:

deck = titanic_df['Cabin'].dropna()


# In[293]:

deck.head()


# In[294]:

levels =[]

for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels, columns=['Cabin'])

cabin_df.head()


# In[295]:

sns.factorplot('Cabin',data=cabin_df, palette='summer',kind="count", x_order=['A','B','C','D','E','F','G'])


# In[296]:

titanic_df.head()


# In[297]:

sns.factorplot('Embarked', data=titanic_df, hue='Pclass',kind='count', x_order=['C', 'S','Q'])


# In[298]:

#Question 3 : Who was alone and who was with family?


# In[299]:

titanic_df.head()


# In[300]:

titanic_df['alone'] = titanic_df.SibSp + titanic_df.Parch


# In[301]:

titanic_df['alone'].loc[(titanic_df['alone'] > 0)] = 'With family'

titanic_df['alone'].loc[(titanic_df['alone'] == 0)] = 'Without family'


# In[302]:

titanic_df.head(10)


# In[303]:

sns.factorplot('alone',data=titanic_df, palette='summer', kind='count')


# In[304]:

sns.factorplot('alone',data=titanic_df, palette='winter', kind='count', hue='Sex')


# In[305]:

sns.factorplot('alone',data=titanic_df, palette='winter', kind='count', hue='person')


# In[306]:

#Question 5: Factors affecting the survivors on the ship?


# In[307]:

titanic_df['Survivor'] = titanic_df['Survived'].map({0:'No',1:'Yes'})


# In[308]:

titanic_df.head()


# In[309]:

sns.factorplot('Survivor', data=titanic_df, kind='count')


# In[310]:

sns.lmplot(x='Age', y='Survived', data=titanic_df)


# In[311]:

sns.lmplot(x='Age', y='Survived', data=titanic_df, hue='Sex')


# In[312]:

sns.lmplot(x='Age', y='Survived', data=titanic_df, hue='Pclass')


# In[313]:

sns.lmplot(x='Age', y='Survived', data=titanic_df, hue='alone', x_bins=[10,20,40,60,80])


# In[314]:

sns.factorplot(x='Pclass', y='Survived', data=titanic_df, hue='Sex')


# In[315]:

sns.factorplot(x='Pclass', y='Survived', data=titanic_df, hue='person')


# In[316]:

sns.factorplot(x='Pclass', y='Survived', data=titanic_df)


# In[317]:

# Additional 2 questions:
# 1. Did the deck have an effect on the passengers survival rate? Did the answermatch up the intuition?
# 2. Did having a family member increase the odds of survival?


# In[318]:

deck.head()


# In[324]:

cabin_df1 = DataFrame(deck)


# In[334]:

#cabin_df1['CL'] = cabin_df1['Cabin'].astype(str).str[0]

cabin_df1['Survivor'] = titanic_df.Survivor
    
cabin_df1.head()


# In[365]:

sns.factorplot('CL', data=cabin_df1, kind='count', order=['A','B','C','D','E','F','G'])


# In[381]:

sns.factorplot('CL', data=cabin_df1, hue='Survivor', kind='count', order=['A','B','C','D','E','F','G'])


# In[392]:

cabin_df1['Survived'] = titanic_df.Survived
sns.factorplot(y='Survived',x='CL', data=cabin_df1, order=['A','B','C','D','E','F','G'])


# In[388]:

sns.factorplot(y='Survived',x='alone', data=titanic_df,hue='person')


# In[ ]:




# In[ ]:



