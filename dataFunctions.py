from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
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
#=====================================================================================
def fillTitles(dframe):
	dframe['Title'] = dframe['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
	myDict = {	"Capt":"Officer","Col":"Officer","Major":"Officer","Dr":"Officer","Rev":"Officer",
				"Lady" :"Royalty","Jonkheer":"Royalty","Don":"Royalty","Sir" :"Royalty",
				"the Countess":"Royalty","Dona":"Royalty",
				"Mme":"Mrs","Mlle":"Miss","Ms":"Mrs","Mr" :"Mr",
				"Mrs":"Mrs","Miss":"Miss","Master":"Master"
			}

	dframe['Title'] = dframe.Title.map(myDict)
	temp_title = pd.get_dummies( dframe['Title'])
	#dframe = dframe.drop('Name', 1)
	status('Processing Title')
	return pd.concat( [ dframe, temp_title ], axis = 1)
#=====================================================================================
def fillTitlesNew(dframe):
	dframe['Title'] = dframe['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
	dframe.loc[dframe["Title"] == "Mlle", "Title"] = 'Miss'
	dframe.loc[dframe["Title"] == "Ms", "Title"] = 'Miss'
	dframe.loc[dframe["Title"] == "Mme", "Title"] = 'Mrs'
	dframe.loc[dframe["Title"] == "Dona", "Title"] = 'Rare Title'
	dframe.loc[dframe["Title"] == "Lady", "Title"] = 'Rare Title'
	dframe.loc[dframe["Title"] == "Countess", "Title"] = 'Rare Title'
	dframe.loc[dframe["Title"] == "Capt", "Title"] = 'Rare Title'
	dframe.loc[dframe["Title"] == "Col", "Title"] = 'Rare Title'
	dframe.loc[dframe["Title"] == "Don", "Title"] = 'Rare Title'
	dframe.loc[dframe["Title"] == "Major", "Title"] = 'Rare Title'
	dframe.loc[dframe["Title"] == "Rev", "Title"] = 'Rare Title'
	dframe.loc[dframe["Title"] == "Sir", "Title"] = 'Rare Title'
	dframe.loc[dframe["Title"] == "Jonkheer", "Title"] = 'Rare Title'
	dframe.loc[dframe["Title"] == "Dr", "Title"] = 'Rare Title'
	status('Processing Title new')
	return dframe
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
	dframe["TicketNumber"] = dframe["Ticket"].str.extract('(\d{2,})', expand=True)
	dframe["TicketNumber"] = dframe["TicketNumber"].apply(pd.to_numeric)
	dframe.TicketNumber.fillna(dframe["TicketNumber"].median(), inplace=True)
	status('Ticket number')
	return dframe
#=====================================================================================
def fillTicketGroup(dframe):
	dframe["Shared_ticket"]=np.where(dframe.groupby("Ticket")["Name"].transform("count") > 1, 1, 0)
	dframe["Ticket_group"]=dframe.groupby("Ticket")["Name"].transform("count")
	status('Ticket Grouping')
	return dframe
#=====================================================================================
# def fillTicket(dframe):
# 	dframe[ 'Ticket' ] = dframe[ 'Ticket' ].map( cleanTicket )
# 	temp_ticket = pd.get_dummies( dframe[ 'Ticket' ] , prefix = 'Ticket' )
# 	dframe = dframe.drop('Ticket', 1)
# 	status('Ticket')
# 	return pd.concat( [ dframe, temp_ticket ], axis = 1)
#=====================================================================================

#=====================================================================================
def fillFare(dframe):
	median_fare=dframe[(dframe["Pclass"] == 3) & (dframe["Embarked"] == 'S')]["Fare"].median()
	dframe["Fare"] = dframe["Fare"].fillna(median_fare)
	dframe["Fare"] = dframe["Fare"].astype(int)
	status("Converting Fare")
	return dframe
#=====================================================================================
def fillRealFare(dframe):
	dframe[ 'Fare_cat' ] = pd.DataFrame(np.floor(np.log10(dframe['Fare' ] + 1))).astype('int')
	#dframe[ 'Bad_ticket' ] = dframe[ 'TicketType' ].isin(['3','4','5','6','7','8','A','L','W'])
	status('Ticket group')
	dframe['Fare_eff'] = dframe['Fare']/dframe['Ticket_group']
	dframe['Fare_eff_cat'] = np.where(dframe['Fare_eff']>16.0, 2, 1)
	dframe['Fare_eff_cat'] = np.where(dframe['Fare_eff']<8.5,0,dframe['Fare_eff_cat'])
	return dframe
#=====================================================================================
def convertSexToNum(dframe,dropSex=True):
	dframe["Female"] = dframe["Sex"].apply(lambda sex: 0 if sex == "male" else 1)
	if dropSex:
		dframe = dframe.drop("Sex",axis=1)
		dframe.columns = dframe.columns.str.replace("Female","Sex")
	status("Converting Sex to num")
	return dframe
#=====================================================================================
def fillDeck(dframe):
	# Naming the Deck accordingly to the Cabin description
	dframe["Deck"] = dframe["Cabin"].str[0]
	# Naming the Deck as U due to unknown Cabin description
	dframe["Deck"] = dframe["Deck"].fillna(value='U')
	status("Deck")
	return dframe
#=====================================================================================
def fillDeckNew(dframe):
	# Naming the Deck accordingly to the Cabin description
	dframe["Deck"] = dframe["Cabin"].str[0]
	# Naming the Deck as U due to unknown Cabin description
	dframe["Deck"] = dframe["Deck"].fillna(value='U')
	dframe["Deck"] = dframe["Deck"].map( lambda c : c[0] )
	# dummy encoding ...
	temp_deck = pd.get_dummie(dframe["Deck"] , prefix = "Dk" )
	dframe = dframe.drop("Deck", 1)
	status("Deck")
	return pd.concat( [ dframe, temp_deck ], axis = 1)
#=====================================================================================
def fillCabin(dframe):
	dframe["Cabin"] = dframe["Cabin"].fillna( 'U' )
	dframe["Cabin"] = dframe["Cabin"].map( lambda c : c[0] )
	# dummy encoding ...
	temp_cabin = pd.get_dummie(dframe["Cabin"] , prefix = "Cabin" )
	dframe = dframe.drop("Cabin", 1)
	status('Cabin dummies')
	return pd.concat( [ dframe, temp_cabin ], axis = 1)
#=====================================================================================
def fillEmbarked(dframe):
	# two missing embarked values - filling them winh thd most frequn ore (S"	dfra"e['Embarked']"iloc["1] = "S"
	# dframe['Embarked'].iloc[829] = "S"
	#temp_embarked=pd.get_dummies(dframe['Embarked'], prefix='Embarked')
	# dframe = pd.get_dummies.concat([dframe,temp_embarked],axis=1)
	# dframe.drop('Embarked',axis=1,inplace=True)
	dframe['Embarked'] = dframe['Embarked'].fillna('S')
	status('Embarked')
	return dframe
#=====================================================================================
def fillEmbarkedDummies(dframe):
	dummies = pd.get_dummies(dframe['Embarked'])
	dummies.drop(['S'], axis=1, inplace=True)
	dframe = dframe.join(dummies)
	return dframe
#=====================================================================================
def fillMissingAge(dframe):
	#Feature set
	age_df = dframe[['Age','Embarked','Fare', 'Parch', 'SibSp',
	         'TicketNumber', 'Title','Pclass','Family',
	         'FsizeD','NameLength',"NlengthD",'Deck']]
	# Split sets into train and test
	train  = age_df.loc[ (dframe.Age.notnull()) ]# known Age values
	test = age_df.loc[ (dframe.Age.isnull()) ]# null Ages
	# All age values are stored in a target array
	y = train.values[:, 0]
	# All the other values are stored in the feature array
	X = train.values[:, 1::]
	# Create and fit a model
	rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
	rtr.fit(X, y)
	# Use the fitted model to predict the missing values
	predictedAges = rtr.predict(test.values[:, 1::])
	# Assign those predictions to the full data set
	dframe.loc[ (dframe.Age.isnull()), 'Age' ] = predictedAges
	dframe['Age'] = dframe['Age'].astype(int)
	return dframe
#=====================================================================================
def fillMissingAgeDrop(dframe):
	 #Feature set
    age_df = dframe[['Age','Fare', 'Parch', 'SibSp',
                 'C', 'Q','Sex']]
    # Split sets into train and test
    train  = age_df.loc[ (dframe.Age.notnull()) ]# known Age values
    test = age_df.loc[ (dframe.Age.isnull()) ]# null Ages
    # All age values are stored in a target array
    y = train.values[:, 0]
    # All the other values are stored in the feature array
    X = train.values[:, 1::]
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])
    # Assign those predictions to the full data set
    dframe.loc[ (dframe.Age.isnull()), 'Age' ] = predictedAges
    return dframe
#=====================================================================================
def classMedian(dframe):
	groupAge = dframe.head(891).groupby(["Sex","Pclass","Title"])
	groupAgeMedian = groupAge.median()
	#grouped_test = combined.iloc[891:].groupby(['Sex','Pclass','Title'])
	#grouped_median_test = grouped_test.median()
	return groupAgeMedian
#=====================================================================================
def fillAge(dframe):
	def getAges(row, ageMedian):
		if row['Sex']==1 and row['Pclass'] == 1:
			if row['Title'] == 'Miss':
				return ageMedian.loc[1, 1,'Miss']['Age']
			elif row['Title'] == 'Mrs':
				return ageMedian.loc[1, 1,'Mrs']['Age']
			elif row['Title'] == 'Officer':
				return ageMedian.loc[1, 1,'Officer']['Age']
			elif row['Title'] == 'Royalty':
				return ageMedian.loc[1, 1,'Royalty']['Age']

		elif row['Sex']==1 and row['Pclass'] == 2:
			if row['Title'] == 'Miss':
				return ageMedian.loc[1, 2,'Miss']['Age']
			elif row['Title'] == 'Mrs':
				return ageMedian.loc[1, 2,'Mrs']['Age']

		elif row['Sex']==1 and row['Pclass'] == 3:
			if row['Title'] == 'Miss':
				return ageMedian.loc[1, 3,'Miss']['Age']
			elif row['Title'] == 'Mrs':
				return ageMedian.loc[1, 3,'Mrs']['Age']

		elif row['Sex']==0 and row['Pclass'] == 1:
			if row['Title'] == 'Master':
				return ageMedian.loc[0, 1,'Master']['Age']
			elif row['Title'] == 'Mr':
				return ageMedian.loc[0, 1,'Mr']['Age']
			elif row['Title'] == 'Officer':
				return ageMedian.loc[0, 1,'Officer']['Age']
			elif row['Title'] == 'Royalty':
				return ageMedian.loc[0, 1,'Royalty']['Age']

		elif row['Sex']==0 and row['Pclass'] == 2:
			if row['Title'] == 'Master':
				return ageMedian.loc[0, 2,'Master']['Age']
			elif row['Title'] == 'Mr':
				return ageMedian.loc[0, 2,'Mr']['Age']
			elif row['Title'] == 'Officer':
				return ageMedian.loc[0, 2,'Officer']['Age']

		elif row['Sex']==0 and row['Pclass'] == 3:
			if row['Title'] == 'Master':
				return ageMedian.loc[0, 3,'Master']['Age']
			elif row['Title'] == 'Mr':
				return ageMedian.loc[0, 3,'Mr']['Age']
	groupAgeMedian = classMedian(dframe)
	dframe['Age'] = dframe.apply(lambda r : getAges(r, groupAgeMedian) if np.isnan(r['Age']) else r['Age'], axis=1)
	#dframe.iloc[891:].Age = dframe.iloc[891:].apply(lambda r : getAges(r, grouped_median_test) if np.isnan(r['Age']) else r['Age'], axis=1)
	#combined["Age"] = combined.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
	status('Age')
	del groupAgeMedian
	return dframe
#=====================================================================================
# def fillMissingFare(dframe):
# 	dframe['Fare'] = dframe['Fare'].fillna( dframe['Fare'].mean() )
# 	status('Fare')
# 	return dframe
#=====================================================================================
def fillNameSize(dframe):
	dframe['NameLength'] = dframe["Name"].apply(lambda x: len(x))
	bins = [0, 20, 40, 57, 85]
	group_names = ['short', 'okay', 'good', 'long']
	dframe['NlengthD'] = pd.cut(dframe['NameLength'], bins, labels=group_names)
	status('Name Size')
	return dframe
#=====================================================================================
def fillFamilySize(dframe):
	dframe[ 'Family' ] = dframe[ 'Parch' ] + dframe[ 'SibSp' ] + 1
	dframe.loc[dframe["Family"] == 1, "FsizeD"] = 'singleton'
	dframe.loc[(dframe["Family"] > 1)  &  (dframe["Family"] < 5) , "FsizeD"] = 'small'
	dframe.loc[dframe["Family"] >4, "FsizeD"] = 'large'
	status('Family Size')
	return dframe
#=====================================================================================
def fillFamily(dframe):
	dframe[ 'Family' ] = dframe[ 'Parch' ] + dframe[ 'SibSp' ] + 1
	dframe[ 'Family' ].loc[dframe["Family"] > 0] = 1
	dframe[ 'Family' ].loc[dframe["Family"] == 0] = 0
	dframe = dframe.drop(['SibSp', 'Parch'], 1)
	status('Family')
	return dframe
#=====================================================================================
def fillChildYoung(dframe):
	dframe[ 'Child'] = dframe['Age']<=10
	dframe[ 'Young' ] = (dframe[ 'Age' ]<=30) | dframe['Age']>10 | (dframe['Title'].isin(['Master','Miss','Mlle']))
	dframe[ 'Adult' ] = dframe['Age']>18 | (dframe['Title'].isin(['Master','Mrs','Mr','Master','Royalty','Offic']))
	return dframe
#=====================================================================================,
def fillPerson(dframe):
	#must run before convertSexToNum
	dframe['Person'] = dframe[['Age','Sex']].apply(get_person,axis=1)
	person_dummies  = pd.get_dummies(dframe['Person'])
	person_dummies.columns = ['Child','Female','Male']
	person_dummies.drop(['Male'], axis=1, inplace=True)
	dframe = dframe.join(person_dummies)
	status('Person')
	return dframe
#=====================================================================================,
def fillPclass(dframe):
	dummies  = pd.get_dummies(dframe['Pclass'])
	dummies.columns = ['Class_1','Class_2','Class_3']
	dummies.drop(['Class_3'], axis=1, inplace=True)
	dframe = dframe.join(dummies)
	return dframe
#=====================================================================================,
def featureScale(dframe):
	std_scale = preprocessing.StandardScaler().fit(dframe[['Age', 'Fare']])
	dframe[['Age', 'Fare']] = std_scale.transform(dframe[['Age', 'Fare']])
	return dframe
#=====================================================================================
def featureEng(dframe):
	# dframe[ 'Family1' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 1 else 0 )
	# dframe[ 'Family2' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 2 else 0 )
	# dframe[ 'Family3' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 3 else 0 )
	# dframe[ 'Family4' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 4 else 0 )
	# dframe[ 'Family5' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 5 else 0 )
	# dframe[ 'Family6' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 6 else 0 )
	# dframe[ 'Family7' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 7 else 0 )
	# dframe[ 'Family8' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 8 else 0 )
	# dframe[ 'Family9' ] = dframe[ 'Family' ].map( lambda s : 1 if s == 9 else 0 )
	# dframe[ 'Family10' ] = dframe[ 'Family' ].map( lambda s : 1 if s >= 10 else 0 )
	#dframe[ 'TicketType' ] = dframe[ 'Ticket' ].str[0]
	#dframe[ 'Title' ] = dframe[ 'Name' ].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
	status('Fare')
	return dframe
#=====================================================================================
def featureProcessing(dframe):
	status('Step 1 : Sex')
	dframe=convertSexToNum(dframe,dropSex=True)
	status('Step 2 : Titles')
	dframe=fillTitles(dframe) #tested and best approach 0.837278401998
	#dframe=fillTitlesNew(dframe)
	status('Step 3 : Deck')
	dframe=fillDeck(dframe)
	status('Step 4 : Ticket')
	dframe=fillTicket(dframe)
	status('Step 5 : Fare')
	dframe=fillFare(dframe)
	status('Step 4_ : Ticket')
	dframe=fillTicketGroup(dframe)
	status('Step 5_ : Fare')
	dframe=fillRealFare(dframe)
	status('Step 6 : Family Size')
	dframe=fillFamilySize(dframe)
	status('Step 7 : Embarked')
	dframe=fillEmbarked(dframe)
	status('Step 9 : Name Size')
	dframe=fillNameSize(dframe)
	status('Step 10: Childhood')
	dframe=fillChildYoung(dframe)
	status('Step 8 : Age')
	#dframe=fillAge(dframe) #0.828302122347
	dframe=categoricalToNum(dframe)
	dframe=fillMissingAge(dframe) #0.829413233458 Accuracy
	status('Preprocessing done !')
	return dframe
#=====================================================================================
def categoricalToNum(dframe):
	labelEnc=LabelEncoder()
	cat_vars=['Embarked','Sex',"Title","FsizeD","NlengthD",'Deck']
	for col in cat_vars:
	    dframe[col]=labelEnc.fit_transform(dframe[col])
	return dframe
#=====================================================================================
def status(message):
	print('Process ',message,' : Concluded!')
#=====================================================================================
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
