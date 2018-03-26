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


def calcScore(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

def runRandomForest(train_x, targets, run_gs = False):
# turn run_gs to True if you want to run the gridsearch again.
#run_gs = False
if run_gs:
	parameter_grid = {
		'max_depth' : [4, 6, 8],
		'n_estimators': [50, 10],
		'max_features': ['sqrt', 'auto', 'log2'],
		'min_samples_split': [1, 3, 10],
		'min_samples_leaf': [1, 3, 10],
		'bootstrap': [True, False],
		}
	forest = RandomForestClassifier()
	cross_validation = StratifiedKFold(targets, n_folds=10)
	grid_search = GridSearchCV(forest, scoring='accuracy', param_grid=parameter_grid, cv=cross_validation)
	grid_search.fit(train, targets)
	model = grid_search
	parameters = grid_search.best_params_
	print('Best score: {}'.format(grid_search.best_score_))
	print('Best parameters: {}'.format(grid_search.best_params_))
else:
	parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 100, 'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
	model = RandomForestClassifier(**parameters)
	model.fit(train, targets)
return calcScore(model, train_X , targets, scoring='accuracy')

def savePredict(model, dframe):
	output = model.predict(dframe).astype(int)
	df_output = pd.DataFrame()
	aux = pd.read_csv('../../data/test.csv')
	df_output['PassengerId'] = aux['PassengerId']
	df_output['Survived'] = output
	df_output[['PassengerId','Survived']].to_csv('../../data/output.csv',index=False)
    
def evaluateModels(train_x, targets):
    resultScores = {'RandomForestClassifier': 0.0,
                    'DecisionTreeClassifier': 0.0,
                    'MLPClassifier': 0.0,
                    'GradientBoostingClassifier': 0.0,
                    'ExtraTreesClassifier': 0.0,
                    'BaggingClassifier': 0.0,
                    'AdaBoostClassifier': 0.0,
                    'LinearSVC': 0.0,
                    'LogisticRegression': 0.0}



	parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 100, 'min_samples_split': 10,
				  'max_features': 'sqrt', 'max_depth': 6}
	model = RandomForestClassifier(**parameters)
	model.fit(train_X, target_y)
	resultScores['RandomForestClassifier'] = calcScore(model, train_X, targets, scoring='accuracy')

	model = DecisionTreeClassifier(random_state=0)
	model.fit( train_X , targets )
    resultScores['DecisionTreeClassifier'] = calcScore(model, train_X , targets, scoring='accuracy')

	model = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(10, 4), random_state=1)
	model.fit(train_X, targets)
    resultScores['MLPClassifier'] = calcScore(model, train_X, targets, scoring='accuracy')

	model = GradientBoostingClassifier()
	model.fit(train_X, targets)
    resultScores['GradientBoostingClassifier'] = calcScore(model, train_X, targets, scoring='accuracy')

	model = ExtraTreesClassifier()
	model.fit(train_X, targets)
    resultScores['ExtraTreesClassifier'] = calcScore(model, train_X, targets, scoring='accuracy')

	model = BaggingClassifier()
	model.fit(train_X, targets)
    resultScores['BaggingClassifier'] = calcScore(model, train_X, targets, scoring='accuracy')

	model = AdaBoostClassifier()
	model.fit(train_X, targets)
    resultScores['AdaBoostClassifier'] = calcScore(model, train_X, targets, scoring='accuracy')

	model = LinearSVC(random_state=0)
	model.fit(train_X, targets)
    resultScores['LinearSVC'] = calcScore(model, train_X, targets, scoring='accuracy')

	model = LogisticRegression()
	model.fit(train_X, targets)
    resultScores['LogisticRegression'] = calcScore(model, train_X, targets, scoring='accuracy')



