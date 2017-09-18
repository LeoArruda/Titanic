
# ### Initial Idea
# 1. Load Library Modules
# 2. Load Datasets
# 3. Explore datasets
# 4. Analyse relations between features
# 5. Analyse missing values
# 6. Analyse features
# 7. Prepare for modelling
# 8. Modelling
# 9. Prepare the prediction for submission

# ### 1. Loading Library Modules



import warnings
warnings.filterwarnings('ignore')

# SKLearn Model Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression , Perceptron

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC

# SKLearn ensemble classifiers
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier , BaggingClassifier
from sklearn.ensemble import VotingClassifier , AdaBoostClassifier

# SKLearn Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# plot functions
import pltFunctions as pfunc

# Configure visualisations
get_ipython().magic('matplotlib inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# ### 2. Loading Datasets

# In[2]:

train = pd.read_csv("./input/train.csv")
test    = pd.read_csv("./input/test.csv")


# In[3]:

#combined = pd.concat([train.drop('Survived',1),test])
#combined = train.append( test, ignore_index = True)
full = train.append( test, ignore_index = True)
del train, test
#train = full[ :891 ]
#combined = combined.drop( 'Survived',1)


# In[4]:

#print ('Datasets:' , 'combined:' , combined.shape , 'full:' , full.shape , 'train:' , train.shape)


# ### 3. Exploring datasets

# In[5]:

full.head(10)


# In[7]:

pd.crosstab(full['Pclass'], full['Sex'])


# In[20]:

print( full.groupby(['Sex','Pclass'])['Age'].mean() )
agedf = full.groupby(['Sex','Pclass'])['Age'].mean()
type( agedf )


# In[25]:

print( full.where((full['Sex'] == 0) & (full['Pclass'] == 1)).groupby(['Pclass','Sex'])['Age'].mean() )
print( full['Sex'].isnull().sum() )


# In[29]:

print( full.where(full['Cabin'].isnull()).groupby(['Name'])['Ticket']. )
full.where(full['Cabin']

# In[15]:

full = pfunc.convertSex(full)
full.head()


# In[17]:

# Naming the Deck accordingly to the Cabin description
full['Deck'] = full['Cabin'].str[0]
# Naming the Deck as U due to unknown Cabin description
full['Deck'] = full['Deck'].fillna(value='U')

pd.crosstab(full['Deck'], full['Survived'])


# In[11]:

print(full.isnull().sum())
print("========================================")
print(train.info())


# In[12]:

#pfunc.pltCorrel( combined )
pfunc.pltCorrel( train )
#pfunc.pltCorrel( full )


# ### Correlations to Investigate
# 
#   __Pclass__ is correlated to __Fare__  ( 1st class tickets would be more expensive than other classes )
#   
#   __Pclass__ x __Age__
#   
#   __SibSp__ X __Age__
# 
#   __SibSp__ x __Fare__
# 
#   __SibSp__ is correlate to __Parch__  ( large families would have high values of parents aboard and solo travellers would have zero parents aboard )
# 
#   __Pclass__ noticeable correlates to __Survived__  ( Expected correlation with higher classes to survive as known ) 

# In[9]:

# Plot distributions of Age of passangers who survived or did not survive
pfunc.pltDistro( train , var = 'Age' , target = 'Survived' , row = 'Sex' )


# In[10]:

# Plot distributions of Fare of passangers who survived or did not survive
pfunc.pltDistro( train , var = 'Survived' , target = 'Pclass' , row = 'Sex' )


# In[11]:

# Plot distributions of Parch of passangers who survived or did not survive
pfunc.pltDistro( train , var = 'Parch' , target = 'Survived' , row = 'Sex' )

# Plot distributions of Age of passangers who survived or did not survive

pfunc.pltCategories( train , cat = 'Embarked' , target = 'Survived' ) 
pfunc.pltCategories( train , cat = 'Pclass' , target = 'Survived' )
pfunc.pltCategories( train , cat = 'Sex' , target = 'Survived' )
pfunc.pltCategories( train , cat = 'Parch' , target = 'Survived' )
pfunc.pltCategories( train , cat = 'SibSp' , target = 'Survived' )
#pfunc.pltDistro( train , var = 'Age' , target = 'Survived' , row = 'Sex' )
