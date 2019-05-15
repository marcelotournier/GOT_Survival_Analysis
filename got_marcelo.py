#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
                          ,     \    /      ,
                         / \    )\__/(     / \
                        /   \  (_\  /_)   /   \
     __________________/_____\__\@  @/___/_____\_________________
     |                          |\../|                          |
     |                           \VV/                           |
     |                                                          |
     |          GAME OF THRONES - SURVIVING WITH DATA!          |
     |           Created on Tue Mar 26 18:28:47 2019            |
     |                                                          |
     |                @author: marcelotournier                  |
     |         Hult International Business School - MBAN        |
     |__________________________________________________________|
                   |    /\ /      \\       \ /\    |
                   |  /   V        ))       V   \  |
                   |/     `       //        '     \|
                   `              V                ' 

#################################################################################
DATA DICTIONARY:

    Variable  	                Description
0  	S.No	                      Character number (by order of appearance)
1   name	                      Character name
2	  title	                      Honorary title(s) given to each character
3	  male	                      1 = male, 0 = female
4	  culture	                    Indicates the cultural group of a character
5	  dateOfBirth	                Known dates of birth for each character (measurement unknown)
6	  mother	                    Character's biological mother
7	  father	                    Character's biological father
8	  heir	                      Character's biological heir
9	  house	                      Indicates a character's allegiance to a house (i.e. a powerful family)
10	spouse	                    Character's spouse(s)
11	book1_A_Game_Of_Thrones	    1 = appeared in book, 0 = did not appear in book
12	book2_A_Clash_Of_Kings	    1 = appeared in book, 0 = did not appear in book
13	book3_A_Storm_Of_Swords	    1 = appeared in book, 0 = did not appear in book
14	book4_A_Feast_For_Crows	    1 = appeared in book, 0 = did not appear in book
15	book5_A_Dance_with_Dragons	1 = appeared in book, 0 = did not appear in book
16	isAliveMother	              1 = alive, 0 = not alive
17	isAliveFather	              1 = alive, 0 = not alive
18	isAliveHeir	                1 = alive, 0 = not alive
19	isAliveSpouse	              1 = alive, 0 = not alive
20	isMarried	                  1 = married, 0 = not married
21	isNoble	                    1 = noble, 0 = not noble
22	age	                        Character's age in years
23	numDeadRelations	          Total number of deceased relatives throughout all books
24	popularity	                Indicates the popularity of a character (1 = extremely popular (max), 0 = extremely unpopular (min))
25	isAlive	                    1 = alive, 0 = not alive

"""

# importing main packages for data manipulation and plotting:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# configuring pandas to display more columns and rows (more useful in Jupyter IDE)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# Loading dataset:
got = pd.read_excel('GOT_character_predictions.xlsx')

# First glance at the data:
got.head()
got.tail()

# dataset shape:
got.shape

# dataset info:
got.info()

# Missing values:
got_missing = got.isna().sum().sort_values()
got_missing = got_missing[got_missing > 0]
print(got_missing)

# Flagging missing values as binary features showing if the information is unknown
for col in got_missing.index:
    got['ukn_'+col] = got[col].isna().astype('int') 
    
# drop columns with NAs
missing_cols = ['name','title','culture','dateOfBirth','mother','father','heir','house',
                'spouse','isAliveMother','isAliveFather','isAliveHeir','isAliveSpouse','age']
got_drop = got.drop(missing_cols,axis=1)

# Get info from non-null variables:
got_drop.info()


########################################################################### 
# Feature Creation Time!

# Variable 'n_books' -> Total number of books in which a character appears
got_drop['n_books'] = got.book1_A_Game_Of_Thrones+got.book2_A_Clash_Of_Kings+got.book3_A_Storm_Of_Swords+got.book4_A_Feast_For_Crows+got.book5_A_Dance_with_Dragons

# Creating binary variable 'noBook' -> 1 = characters who aren't flagged as present in any GOT book
got_drop['noBook'] = (got_drop.n_books == 0).astype('int')

# Creating binary variable 'noDeadRelations' -> 1 = characters without dead relations
got_drop['noDeadRelations'] = (got.numDeadRelations == 0).astype('int')

# Creating binary variable 'hiDeadRelations' -> 1 = characters with 4+ dead relations
got_drop['hiDeadRelations'] = (got.numDeadRelations >= 4).astype('int')

# Creating binary variable 'lowPopularity' -> 1 = characters with popularity < 0.05
got_drop['lowPopularity'] = (got.popularity < 0.05).astype('int') # original < 0.1

# Creating binary variable 'hiPopularity' -> 1 = characters with popularity > 0.6
got_drop['hiPopularity'] = (got.popularity >= 0.6).astype('int')

# Creating a feature 'unknown level' -> total sum of NAs in the character's row
got_drop['ukn_level'] = got_drop.ukn_house+ got_drop.ukn_title+ got_drop.ukn_culture+ got_drop.ukn_dateOfBirth+ got_drop.ukn_age+ got_drop.ukn_spouse+ got_drop.ukn_isAliveSpouse+ got_drop.ukn_father+ got_drop.ukn_isAliveFather+ got_drop.ukn_heir+ got_drop.ukn_isAliveHeir+got_drop.ukn_mother+ got_drop.ukn_isAliveMother 

# Creating binary variable 'low_ukn' -> 1 = characters with 'ukn_level' <= 6
got_drop['low_ukn'] = (got_drop.ukn_level <= 6).astype('int')


##### FEATURES FOR HOUSES & CULTURES #####

# Filling NAs in got['house'] with the string 'Missing'
got['house'] = got.house.fillna('Missing')

# Creating a lambda function to calculate how many persons belong to the character's house:
house_size = lambda house: got.loc[got.house == house,'S.No'].count()

# applying lambda function to new column - house size:
got_drop['houseSize'] = got['house'].apply(house_size)

# correcting values of 'Missing' house size to zero:
got_drop.loc[got_drop['houseSize'] == 427,'houseSize'] = 0

# Creating a lambda function to calculate a house's popularity - which will be the mean of popularity of it's members:
house_pop = lambda house: got.loc[got.house == house,'popularity'].mean()

# applying lambda function to new column - house popularity:
got_drop['housePopularity'] = got['house'].apply(house_pop)

# Creating binary variable 'hiHousePop' -> 1 = characters with housePopularity >= 0.4
got_drop['hiHousePop'] = (got_drop.housePopularity >= 0.4).astype('int')

# Filling NAs in got['culture'] with the string 'Missing'
got['culture'] = got.culture.fillna('Missing')

# Creating a lambda function to calculate how many persons belong to the character's culture:
culture_size = lambda culture: got.loc[got.culture == culture,'S.No'].count()

# applying lambda function to new column - culture size:
got_drop['cultureSize'] = got['culture'].apply(culture_size)

# Creating a lambda function to calculate a culture's popularity - which will be the mean of popularity of it's members:
culture_pop = lambda culture: got.loc[got.culture == culture,'popularity'].mean()

# applying lambda function to new column - culture popularity:
got_drop['culturePopularity'] = got['culture'].apply(culture_pop)

# Creating variables accordingly to house size:
got_drop['one-pers_house'] = (got_drop.houseSize == 1).astype('int') # last in house
got_drop['sm_med_house'] = ((got_drop.houseSize >= 2)&(got_drop.houseSize < 20)).astype('int') # small to medium house
got_drop['big_house'] = ((got_drop.houseSize >= 20)&(got_drop.houseSize < 100)).astype('int') # big house
got_drop['huge_house'] = ((got_drop.houseSize >= 100)&(got_drop.houseSize < 200)).astype('int') # huge house

# Creating variables accordingly to culture size:
got_drop['one-pers_culture'] = (got_drop.cultureSize == 1).astype('int') # last in house
got_drop['sm_culture'] = ((got_drop.cultureSize >= 2)&(got_drop.cultureSize < 5)).astype('int') # small culture
got_drop['med_culture'] = ((got_drop.cultureSize >= 5)&(got_drop.cultureSize < 20)).astype('int') # medium culture
got_drop['big_culture'] = ((got_drop.cultureSize >= 20)&(got_drop.cultureSize < 100)).astype('int') # big culture
got_drop['huge_culture'] = ((got_drop.cultureSize >= 100)&(got_drop.cultureSize < 200)).astype('int') # huge culture



# Saving dataset to further analysis and modelling:
got_drop.to_excel('got_features.xlsx')

#########################################################
# EXPLORATORY DATA ANALYSIS

# Let's look at the frequency distributions for our variables:

# with histograms:
got_drop.hist(figsize=(22,18))
plt.savefig('histograms.png')
plt.show()

# A further look on the distribution of continuous variables with boxplots:

for var in ['S.No','n_books','numDeadRelations','houseSize','housePopularity','cultureSize','culturePopularity',
            'n_books','popularity','ukn_level']:
    plt.boxplot(got_drop[var],vert=False)
    plt.title(var)
    plt.show()

# comparing the distribution of some variables between alive and dead characters:
for var in ['n_books','numDeadRelations','popularity','n_books','houseSize']:
    sns.boxplot(data=got_drop,x='isAlive',y=got_drop[var])
    plt.show()

# Looking at a summary of descriptive statistics:
got_drop.describe().round(2)

# Summary statistics of dead and alive character subgroups:
dead = got.loc[got.isAlive == 0,:]
alive = got.loc[got.isAlive == 1,:]

dead_summary = dead.describe().round(2)
alive_summary = alive.describe().round(2)

print(dead_summary) # as I used Jupyter for the analysis, I didn't used print (just 'dead_summary')

print(alive_summary) # as I used Jupyter for the analysis, I didn't used print (just 'alive_summary')

# Analyzing variable correlations with survivability:
got_corr = got_drop.corr()['isAlive'].drop('isAlive').sort_values(ascending=False)
print('# variable correlation with survival: #\n')
print(got_corr)

# Looking at a pivot table aggregating totals by 'isAlive':

table_totals = pd.pivot_table(got_drop,index='isAlive',aggfunc=np.count_nonzero)
print(table_totals) # as I used Jupyter for the analysis, I didn't used print (just 'table_totals')

#### Exploring houses:

table_houses = pd.pivot_table(got,index='house',aggfunc=np.count_nonzero)
#.sort_values('S.No',ascending=False)
pop_houses = pd.pivot_table(got,index='house',aggfunc=np.mean)
#.sort_values('S.No',ascending=False)
table_houses = pd.concat([table_houses.loc[:,['S.No','isAlive']],pop_houses.loc[:,'popularity']],axis=1,sort=True).sort_values('S.No',ascending=False)
table_houses['pctSurvival'] = (table_houses['isAlive']/table_houses['S.No']).round(2)
table_houses = table_houses.loc[:,:].sort_values([
    
    'S.No',
    'pctSurvival',
    'popularity'
],
    ascending=False
)

print(table_houses) # as I used Jupyter for the analysis, I didn't used print (just 'table_houses')


##### Are there patterns to look for survivability?

# function for adding jitter to better visualize data in scatterplots:
def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

plt.scatter(table_houses['popularity'],rand_jitter(table_houses.pctSurvival))
plt.show()
#sns.lmplot(x='houseSize',y='housePopularity', hue='isAlive',data=got_drop,x_jitter=0.1,y_jitter=0.1)

plt.scatter(table_houses['S.No'],rand_jitter(table_houses.popularity))
plt.show()

print(table_houses.loc[(table_houses['popularity'] < 0.04)&(table_houses['S.No'] < 10),'pctSurvival'].mean()) # small unpopular houses average

print(table_houses.loc[(table_houses['popularity'] < 0.04)&(table_houses['S.No'] < 10),'S.No'].sum()) # small unpopular houses count

# What are the survival percentages accordingly to house sizes?
print('one-person house:',table_houses.loc[(table_houses['S.No'] == 1),'pctSurvival'].mean()) # last in house
print('small & medium houses:',table_houses.loc[(table_houses['S.No'] >= 2)&(table_houses['S.No'] < 20),'pctSurvival'].mean()) # small-medium house
print('big house:',table_houses.loc[(table_houses['S.No'] >= 20)&(table_houses['S.No'] < 100),'pctSurvival'].mean()) # big house
print('huge house:',table_houses.loc[(table_houses['S.No'] >= 100)&(table_houses['S.No'] < 200),'pctSurvival'].mean()) # huge house


#### Exploring cultures:

table_cultures = pd.pivot_table(got,index='culture',aggfunc=np.count_nonzero)
#.sort_values('S.No',ascending=False)
pop_cultures = pd.pivot_table(got,index='culture',aggfunc=np.mean)
#.sort_values('S.No',ascending=False)
table_cultures = pd.concat([table_cultures.loc[:,['name','isAlive']],
                            pop_cultures.loc[:,'popularity'],
                           pop_cultures.loc[:,'S.No']],
                           axis=1,sort=True).sort_values('S.No',ascending=False)
table_cultures['pctSurvival'] = (table_cultures['isAlive']/table_cultures['name']).round(2)
table_cultures = table_cultures.loc[:,:].sort_values(['S.No','pctSurvival','name','popularity'],
                                                           ascending=False)
 
print(table_cultures) 
                    # as I used Jupyter for the analysis, I didn't used print (just 'table_cultures')

                                                     
##### Are there patterns to look for survivability?                                                     
plt.scatter(got_drop.cultureSize,rand_jitter(got_drop.isAlive))
plt.show()
#plt.scatter(got_drop.cultureSize,got_drop.culturePopularity)
sns.lmplot(x='cultureSize',y='culturePopularity', hue='isAlive',data=got_drop,x_jitter=0.1,y_jitter=0.1)
plt.show()

# What are the survival percentages accordingly to culture sizes?    
print('one-person culture:',table_cultures.loc[(table_cultures.name == 1),'pctSurvival'].mean()) # last in culture
print('small culture:',table_cultures.loc[(table_cultures.name >= 2)&(table_cultures.name < 5),'pctSurvival'].mean()) # small culture
print('medium culture:',table_cultures.loc[(table_cultures.name >= 5)&(table_cultures.name < 20),'pctSurvival'].mean()) # medium culture
print('big culture:',table_cultures.loc[(table_cultures.name >= 20)&(table_cultures.name < 100),'pctSurvival'].mean()) # big culture
print('huge culture:',table_cultures.loc[(table_cultures.name >= 100)&(table_cultures.name < 200),'pctSurvival'].mean()) # huge culture
                                                     
#How many characters are alive, accordingly to 'ukn_level'?
table_ukn = pd.pivot_table(got_drop,index='ukn_level',aggfunc=np.sum)
table_ukn.loc[:,'isAlive']

#How many characters are alive, accordingly to 'n_books'?                         
table_books = pd.pivot_table(got_drop,index='n_books',aggfunc=np.sum)
table_books.loc[:,'isAlive']
                                                     
#How many characters are alive, accordingly to 'houseSize'?
table_hsize = pd.pivot_table(got_drop,index='houseSize',aggfunc=np.sum)
table_hsize.loc[:,'isAlive']

#are there patterns related to date of birth?                                                     
got_dob = got.loc[got.dateOfBirth.notna(),['dateOfBirth','isAlive']].sort_values('dateOfBirth')
print(got_dob) # as I used Jupyter for the analysis, I didn't used print (just 'got_dob')

# problems with two inconsistent dateOfBirth values - consider removing dates (or rows) from dataset??????
print(got.loc[110,:])
print(got.loc[1350,:])

# No characters alive before DOB 207 -> good predictor?
got_dob.loc[got_dob.dateOfBirth <= 207,'isAlive'].mean()

# how many characters?
got_dob.loc[got_dob.dateOfBirth <= 207,'isAlive'].count()

# 91% of characters alive after DOB 281 -> good predictor?
got_dob.loc[got_dob.dateOfBirth > 281,'isAlive'].mean()

# how many characters?
got_dob.loc[got_dob.dateOfBirth > 281,'isAlive'].count()

# Looking at age with DOB and survivability:
got_age = got.loc[got.age.notna(),['age','dateOfBirth','isAlive']].sort_values('dateOfBirth')
got_age



#################################################################################
# PREDICTIVE MODELLING:

# BASIC ASSUMPTION => BUILDING SIMPLE AND EXPLAINABLE MODELS WITH THE BEST
#                     POSSIBLE RESULTS

# Best approach    => KNN!
# 

# loading scikitlearn modules:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

""" 
Creating a function to accelerate the predictive pipeline:
Load features and target >> scale features >> train-test split >> 
GridSearchCV tuning >> returning optimal parameters and results 
""" 

def test_knn(vars):
    # Load the GOT dataset as features and labels
    X = got_drop.loc[:,vars]
    y = got_drop['isAlive']

    scaler = StandardScaler()


    # Fitting the scaler with our data
    scaler.fit(X)

    # Transforming our data after fit
    X_scaled = scaler.transform(X)
    
    # Train test split
    X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.1,random_state=508)
        
    # Instantiate and fit KNN model with neighbors from 1-20
    
    # Creating a hyperparameter grid
    nneighbors = pd.np.arange(3, 15)
    knn_weights = [
        'distance',
                   'uniform'
                  ]
    #knn_alg = ['auto','ball_tree','kd_tree','brute']
    pval = np.arange(1,2,1)
    param_grid = {'n_neighbors' : nneighbors,
                 'weights':knn_weights,
                 #'algorithm':knn_alg,
                 'p':pval}


    # Building the model object one more time
    knn = KNeighborsClassifier()



    # Creating a GridSearchCV object
    knn_cv = GridSearchCV(knn, param_grid,cv=3,scoring= 'roc_auc')



    # Fit it to the training data
    knn_cv.fit(X_train, y_train)



    # Print the optimal parameters and best score\
    #print('\n\n### KNN ###')
    #print("Tuned KNN Parameter:", knn_cv.best_params_)
    
    nn = knn_cv.best_params_['n_neighbors']
    kw = knn_cv.best_params_['weights']
    #ka = knn_cv.best_params_['algorithm']
    pv = knn_cv.best_params_['p']
    
    knn_best = KNeighborsClassifier(n_neighbors=nn,weights=kw,
                                    #algorithm=kw,
                                    p=pv)
    knn_best.fit(X_train, y_train)
    
    knn_tr_score = knn_best.score(X_train,y_train)
    knn_te_score = knn_best.score(X_test,y_test)
    
    #cv_score = roc_auc_score(y_test, knn_best.predict_proba(X_test)[:,1])

    knn_score = cross_val_score(knn_best,
                           X_scaled,
                           y,
                           cv = 3, scoring= 'roc_auc')


    

    mean_auc = pd.np.mean(knn_score).round(3)
    
    #print("Tuned test KNN Accuracy:", knn_cv.best_score_.round(4))
    return(nn,kw,pv,knn_tr_score,knn_te_score,mean_auc)

# Testing the model with selected variables:

best_variables = ['book1_A_Game_Of_Thrones',
    'book4_A_Feast_For_Crows',
          'popularity',
          'n_books',
          'ukn_level',
          'ukn_title',
    'huge_culture',
    'isMarried',
    'noDeadRelations']

test_knn(best_variables)

# Output: 14 neighbors,weights=uniform ,p=1 , train score = 0.827, test score = 0.820 , mean crossval roc_auc score = 0.802
# (14, 0.8275271273557967, 0.82051282051282048, 0.80200000000000005)






##################################################
# EXTRACTING PREDICTIONS FROM BEST MODEL (KNN):

got_data = got_drop.loc[:,best_variables]
    
got_target = got_drop.loc[:,'isAlive']

scaler = StandardScaler()


# Fitting the scaler with our data
scaler.fit(got_data)

# Transforming our data after fit
got_scaled = scaler.transform(got_data)

X0_train, X0_test, y0_train, y0_test = train_test_split(
                                                        got_scaled,got_target, 
                                                        test_size = 0.1, 
                                                        random_state=508)

knn_final = KNeighborsClassifier(n_neighbors=14,weights='uniform',
                                    #algorithm=,
                                    p=1)
knn_final.fit(X0_train, y0_train)

y0_pred = knn_final.predict(X0_test)



knn_score = cross_val_score(knn_final,
                           got_scaled,
                           got_target,
                           cv = 3, scoring= 'roc_auc')


    
print('final train score:',knn_final.score(X0_train,y0_train))
print('final test score:',knn_final.score(X0_test,y0_test))
print('final roc auc score:',pd.np.mean(knn_score).round(3))

final_predictions = pd.DataFrame(y0_pred)

final_predictions.to_excel('predictions.xlsx')

#y0_test.to_excel('test_labels.xlsx') # use to compare train test results outside python, if needed

#################################
# STUDYING MODEL RESULTS+METRICS:

# building a confusion matrix

conf_mat = confusion_matrix(y_true = y0_test,
                       y_pred = y0_pred)

print(conf_mat)


# Plotting confusion matrix
labels = ['dead', 'alive']

cm = confusion_matrix(y_true = y0_test,
                      y_pred = y0_pred)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'Blues')


plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Confusion matrix of the classifier')
plt.show()

# Plotting more model metrics:
print('true positives:',conf_mat[1,1])
print('true negatives:',conf_mat[0,0])
print('false positives:',conf_mat[0,1])
print('false negatives:',conf_mat[1,0])
print('precision:',(conf_mat[1,1]/(conf_mat[1,1]+conf_mat[0,1])))
print('sensitivity:',(conf_mat[1,1]/(conf_mat[1,1]+conf_mat[1,0])))
print('specificity:',(conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])))

#######
# ROC AUC plot analysis: 

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

probs = knn_final.predict_proba(X0_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y0_test, preds)
roc_auc = auc(fpr, tpr)


# plotting the curve:
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

