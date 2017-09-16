# Titanic
Kaggle dataset

## Competition Description

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

## Data Description

### Overview
The data has been split into two groups:
training set (train.csv)
test set (test.csv)

The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

### Data Dictionary

| __Variable__       |Definition                               |    __Key__                      |
| -------------- | --------------------------------------- | --------------------------- |
| survival       | Survival                                |  0 = No, 1 = Yes            |
| pclass         | Ticket class                            |  1 = 1st, 2 = 2nd, 3 = 3rd  |
| sex            | Sex                                     |                             |
| Age            | Age in years                            |                             |
| sibsp          | # of siblings / spouses aboard the Titanic |                             |
| parch          | # of parents / children aboard the Titanic |                             |
| ticket         | Ticket number                           |                             |
| fare           | Passenger fare                          |                             |
| cabin          | Cabin number                            |                             |
| embarked       | Port of Embarkation                     | C = Cherbourg, Q = Queenstown, S = Southampton |


### Variable Notes

#### pclass A proxy for socio-economic status (SES)__
    1st = Upper
    2nd = Middle
    3rd = Lower

#### age - Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5__

#### sibsp - The dataset defines family relations in this way...

    Sibling = brother, sister, stepbrother, stepsister

    Spouse = husband, wife (mistresses and fiancés were ignored)

#### parch - The dataset defines family relations in this way...

    Parent = mother, father

    Child = daughter, son, stepdaughter, stepson
      
      *Some children travelled only with a nanny, therefore __parch=0 for them.


### Data Preparation
[ ] Apply the proper sex missing value accordingly to name Title
[ ] Update missing value for Cabin accordingly to the Ticket number
[ ] Update missing value for Cabin if some parent has Cabin information


### Feature Engineering

[X] Convert Sex from text to Numeric
[ ] Convert Embarked from text to Numeric

[ ] Pack the families in groups (Same cabin, same lastname,...)

[ ] 
