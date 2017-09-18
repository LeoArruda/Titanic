from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def loadData():
	train = pd.read_csv("./input/train.csv")
	test    = pd.read_csv("./input/test.csv")
	#combined = pd.concat([train.drop('Survived',1),test])
	combined = train.append( test, ignore_index = True)
	del train, test
	#train = full[ :891 ]
	#combined = combined.drop( 'Survived',1)
	return combined
	
def fillTitles(dframe):
	dframe['Title'] = dframe['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
	myDict = {	"Capt":       "Officer", 
	"Col":        "Officer",
	"Major":      "Officer",
	"Dr":         "Officer",
	"Rev":        "Officer",
	"Lady" :      "Royalty",
	"Jonkheer":   "Royalty",
	"Don":        "Royalty",
	"Sir" :       "Royalty",
	"the Countess":"Royalty",
	"Dona":       "Royalty",
	"Mme":        "Mrs",
	"Mlle":       "Miss",
	"Ms":         "Mrs",
	"Mr" :        "Mr",
	"Mrs" :       "Mrs",
	"Miss" :      "Miss",
	"Master" :    "Master"
	}
	dframe['Title'] = dframe.Title.map(myDict)
	temp_title = pd.get_dummies( dframe['Title'])
	#dframe = dframe.drop('Name', 1)
	status('Title')
	return pd.concat( [ dframe, temp_title ], axis = 1)
#=====================================================================================
	
def cleanTicket( ticket ):
	ticket = ticket.replace( '.' , '' )
	ticket = ticket.replace( '/' , '' )
	ticket = ticket.split()
	ticket = map( lambda t : t.strip() , ticket )
	ticket = list(filter( lambda t : not t.isdigit() , ticket ))
	if len( ticket ) > 0:
		return ticket[0]
	else: 
		return 'XXX'
#=====================================================================================

def fillTicket(dframe):
	dframe[ 'Ticket' ] = dframe[ 'Ticket' ].map( cleanTicket )
	temp_ticket = pd.get_dummies( dframe[ 'Ticket' ] , prefix = 'Ticket' )
	dframe = dframe.drop('Ticket', 1)
	status('Ticket')
	return pd.concat( [ dframe, temp_ticket ], axis = 1)
#=====================================================================================

def convertSexToNum(dframe ,dropSex=True):
	dframe["Female"] = dframe["Sex"].apply(lambda sex: 0 if sex == "male" else 1)
	if dropSex == True: 
		dframe = dframe.drop('Sex',axis=1)
		dframe.columns = dframe.columns.str.replace('Female','Sex')
	status('Sex')
	return dframe
#=====================================================================================

def fillDeck(dframe):
	# Naming the Deck accordingly to the Cabin description
	dframe['Deck'] = dframe['Cabin'].str[0]
	# Naming the Deck as U due to unknown Cabin description
	dframe['Deck'] = dframe['Deck'].fillna(value='U')
	status('Deck')
	return dframe
#=====================================================================================

def fillCabin(dframe):
	dframe[ 'Cabin' ] = dframe['Cabin'].fillna( 'U' )
	dframe[ 'Cabin' ] = dframe[ 'Cabin' ].map( lambda c : c[0] )
	# dummy encoding ...
	temp_cabin = pd.get_dummies( dframe['Cabin'] , prefix = 'Cabin' )
	dframe = dframe.drop('Cabin', 1)
	status('Cabin')
	return pd.concat( [ dframe, temp_cabin ], axis = 1)
#=====================================================================================

#http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
def fillEmbarked(dframe):
	# two missing embarked values - filling them with the most frequent one (S)	dframe['Embarked'].iloc[61] = "S"
	dframe['Embarked'].iloc[829] = "S" 
	temp_embarked = pd.get_dummies(dframe['Embarked'], prefix='Embarked')
	dframe = pd.concat([dframe,temp_embarked],axis=1)
	dframe.drop('Embarked',axis=1,inplace=True)
	status('Embarked')
	return dframe
#=====================================================================================
	
def fillMissingAge(dframe):
	dframe['Age'] = dframe['Age'].fillna( dframe['Age'].mean())
	status('Simple Age')
	return dframe
#=====================================================================================
def classMedian(dframe):
	groupAge = dframe.head(891).groupby(['Sex','Pclass','Title'])
	groupAgeMedian = groupAge.median()
	#grouped_test = combined.iloc[891:].groupby(['Sex','Pclass','Title'])
	#grouped_median_test = grouped_test.median()
	return groupAgeMedian

def fillAge(dframe):
	def getAges(row, ageMedian):
		if row['Sex']==1 and row['Pclass'] == 1:
			if row['Title'] == 'Miss':
				return ageMedian.loc[1, 1, 'Miss']['Age']
			elif row['Title'] == 'Mrs':
				return ageMedian.loc[1, 1, 'Mrs']['Age']
			elif row['Title'] == 'Officer':
				return ageMedian.loc[1, 1, 'Officer']['Age']
			elif row['Title'] == 'Royalty':
				return ageMedian.loc[1, 1, 'Royalty']['Age']

		elif row['Sex']==1 and row['Pclass'] == 2:
			if row['Title'] == 'Miss':
				return ageMedian.loc[1, 2, 'Miss']['Age']
			elif row['Title'] == 'Mrs':
				return ageMedian.loc[1, 2, 'Mrs']['Age']

		elif row['Sex']==1 and row['Pclass'] == 3:
			if row['Title'] == 'Miss':
				return ageMedian.loc[1, 3, 'Miss']['Age']
			elif row['Title'] == 'Mrs':
				return ageMedian.loc[1, 3, 'Mrs']['Age']

		elif row['Sex']==0 and row['Pclass'] == 1:
			if row['Title'] == 'Master':
				return ageMedian.loc[0, 1, 'Master']['Age']
			elif row['Title'] == 'Mr':
				return ageMedian.loc[0, 1, 'Mr']['Age']
			elif row['Title'] == 'Officer':
				return ageMedian.loc[0, 1, 'Officer']['Age']
			elif row['Title'] == 'Royalty':
				return ageMedian.loc[0, 1, 'Royalty']['Age']

		elif row['Sex']==0 and row['Pclass'] == 2:
			if row['Title'] == 'Master':
				return ageMedian.loc[0, 2, 'Master']['Age']
			elif row['Title'] == 'Mr':
				return ageMedian.loc[0, 2, 'Mr']['Age']
			elif row['Title'] == 'Officer':
				return ageMedian.loc[0, 2, 'Officer']['Age']

		elif row['Sex']==0 and row['Pclass'] == 3:
			if row['Title'] == 'Master':
				return ageMedian.loc[0, 3, 'Master']['Age']
			elif row['Title'] == 'Mr':
				return ageMedian.loc[0, 3, 'Mr']['Age']
	groupAgeMedian = classMedian(dframe)	
	dframe['Age'] = dframe.apply(lambda r : getAges(r, groupAgeMedian) if np.isnan(r['Age']) else r['Age'], axis=1)
	#dframe.iloc[891:].Age = dframe.iloc[891:].apply(lambda r : getAges(r, grouped_median_test) if np.isnan(r['Age']) else r['Age'], axis=1)
	#combined["Age"] = combined.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
	status('Age')
	del groupAgeMedian
	return dframe
#=====================================================================================

def fillMissingFare(dframe):
	dframe['Fare'] = dframe['Fare'].fillna( dframe['Fare'].mean() )
	status('Fare')
	return dframe
#=====================================================================================
	
def featureEng(dframe):
	dframe[ 'Child'] = dframe['Age']<=10
	dframe[ 'Young' ] = (dframe[ 'Age' ]<=30) | dframe['Age']>10 | (dframe['Title'].isin(['Master','Miss']))
	#dframe[ 'Cabin_known'] = dframe['Cabin'].isnull() == False
	#dframe[ 'Age_known'] = dframe['Age'].isnull() == False
	dframe[ 'Family' ] = dframe[ 'Parch' ] + dframe[ 'SibSp' ] + 1
	dframe[ 'Family1' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 1 else 0 )
	dframe[ 'Family2' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 2 else 0 )
	dframe[ 'Family3' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 3 else 0 )
	dframe[ 'Family4' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 4 else 0 )
	dframe[ 'Family5' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 5 else 0 )
	dframe[ 'Family6' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 6 else 0 )
	dframe[ 'Family7' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 7 else 0 )
	dframe[ 'Family8' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 8 else 0 )
	dframe[ 'Family9' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 9 else 0 )
	dframe[ 'Family10' ] = dframe[ 'Family' ].map( lambda s : 1 if s >= 10 else 0 )
	status('Family')
	#dframe[ 'Deck' ] = dframe['Cabin' ].str[0]
	#dframe[ 'Deck' ] = dframe['Deck' ].fillna(value='U')
	dframe[ 'TicketType' ] = dframe[ 'Ticket' ].str[0]
	#dframe[ 'Title' ] = dframe[ 'Name' ].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
	dframe[ 'Fare_cat' ] = pd.DataFrame(np.floor(np.log10(dframe['Fare' ] + 1))).astype('int')
	#dframe[ 'Bad_ticket' ] = dframe[ 'TicketType' ].isin(['3','4','5','6','7','8','A','L','W'])
	#dframe[ 'Young' ] = (dframe[ 'Age' ]<=30) | dframe['Age']>10 | (dframe['Title'].isin(['Master','Miss','Mlle']))
	dframe[ 'Shared_ticket' ] = np.where(dframe.groupby('Ticket')[ 'Name' ].transform('count') > 1, 1, 0)
	dframe[ 'Ticket_group' ] = dframe.groupby('Ticket')[ 'Name' ].transform('count')
	status('Ticket group')
	dframe[ 'Fare_eff' ] = dframe[ 'Fare' ]/dframe[ 'Ticket_group' ]
	dframe[ 'Fare_eff_cat' ] = np.where(dframe[ 'Fare_eff' ]>16.0, 2, 1)
	dframe[ 'Fare_eff_cat' ] = np.where(dframe[ 'Fare_eff' ]<8.5,0,dframe['Fare_eff_cat'])
	status('Fare')
	return dframe
#=====================================================================================

def status(message):
	print('Process ',message,' : Concluded!')

	