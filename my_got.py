# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:50:54 2019

@author: Moka
"""


# Importing new libraries
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects

# Importing other libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train/test split
from sklearn.model_selection import cross_val_score # k-folds cross validation

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, KFold, train_test_split 
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve 
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 

# Loading Libraries

import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects

from sklearn.metrics import confusion_matrix # confusion matrix
import seaborn as sns                        # visualizing the confusion matrix
from sklearn.metrics import roc_auc_score    # AUC value

#This is to fix "GraphViz executable not found
import os     
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


file = 'GOT_character_predictions.xlsx'

got = pd.read_excel(file)


                   ###########################
                   #Exploratory data analysis#
                   ###########################

# Descriptive statistics
got.describe().round(2)


###############################################################################
# Imputing Missing Values
###############################################################################

print(
      got
      .isnull()
      .sum()
      )


for col in got:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if got[col].isnull().any():
        got['m_'+col] = got[col].isnull().astype(int)

# Exploring age
print(got["age"].mean())
got["age"].describe()
got["age"].sort_values()


# Check which characters have a negative age and it's value.
print(got["name"][got["age"] == -298001])
print(got["name"][got["age"] == -277980])

# Replace negative ages after looking them up in the internet
got.loc[110, "age"] = 0.0
got.loc[1350, "age"] = 25

#Fill the missing values
###Age 

got['title'] = got['title'].fillna('unknown_title')
got['culture']=got['culture'].fillna('unknown_culture')
got['house'] = got['house'].fillna('unknown_house')
got['mother']=got['mother'].fillna('unknown_mother')
got['father']=got['father'].fillna('unknown_father')
got['heir']=got['heir'].fillna('unknown_heir')
got['house']=got['house'].fillna('unknown_house')
got['spouse']=got['spouse'].fillna('unknown_spouse')

fill=got['age'].mean()
got['age']=got['age'].fillna(fill)

# Some nans values are nan because we dont know them so fill them with -1
got.fillna(value=-1, inplace=True)


# Get all of the culture values in our dataset
set(got['culture'])
set(got['house'])

# Lots of different names for one culture so lets group them up
cult = {
    'Summer Islands': ['Summer Islands', 'Summer Islander', 'Summer Isles'],
    'Ghiscari': ['Ghiscari', 'Ghiscaricari'],
    'Asshai': ["Asshai'i", 'Asshai'],
    'Andal': ['Andal','Andals'],
    'Lysene': ['Lysene', 'Lyseni'],
    'Astapor': ['Astapor', 'Astapori'],
    'Braavosi': ['Braavosi', 'Braavos'],
    'Dornish': ['Dornishmen', 'Dorne', 'Dornish'],
    'Lhazareen': ['Lhazareen', 'Lhazarene'],
    'Westermen': ['Westermen', 'Westerman', 'Westerlands', 'westermen'],
    'Stormlander': ['Stormlands', 'Stormlander'],
    'Norvoshi': ['Norvos', 'Norvoshi'],
    'Northmen': ['Northern mountain clans', 'Northmen'],
    'Free Folk': ['Wildling', 'Wieldings', 'First men', 'free folk','Free Folk'],
    'Qartheen': ['Qartheen', 'Qarth'],
    'Reach': ['Reach', 'The Reach', 'Reachmen'],
    'Rivermen': ['Rivermen', 'Riverlands'],
    'Ironborn': ['Ironborn', 'Ironmen'],
    'Mereen': ['Meereen', 'Meereenese'],
    'RiverLands': ['riverlands', 'rivermen'],
    'Vale': ['Vale', 'Valemen', 'Vale mountain clans']
}


def get_cult(value):
    value = value.lower()
    v = [k for (k, v) in cult.items() if value in v]
    return v[0] if len(v) > 0 else value.title()
got.loc[:, "culture"] = [get_cult(x) for x in got["culture"]]


#Grouping the similar houses
hous = {
    'Brotherhood Without Banners': ['Brotherhood Without Banners', 'Brotherhood without Banners', 'Brotherhood without banners']
    }

def get_house(value):
    value = value.lower()
    v = [k for (k, v) in hous.items() if value in v]
    return v[0] if len(v) > 0 else value.title()
got.loc[:, "house"] = [get_house(x) for x in got["house"]]

########################
# Visual EDA (Histograms)
########################

######################## Plotting age ########################################
plt.subplot(2, 2, 1)
sns.distplot(got['age'],
             bins = 35,
             color = 'g')

plt.xlabel('age')


######################## Plotting popularity #################################
plt.subplot(2, 2, 2)
sns.distplot(got['popularity'],
             bins = 30,
             color = 'y')

plt.xlabel('popularity')

######################## Plotting dateOfBirth #################################
plt.subplot(2, 2, 2)
sns.distplot(got['dateOfBirth'],
             bins = 30,
             color = 'y')

plt.xlabel('dateOfBirth')


#Dropping some columns because they have so many missing values or they dont 
#seem to be helpful in the analysis
got1 = got.drop(columns=['S.No', 'name', 'dateOfBirth',
                            'mother', 'father', 'heir',
                            'isAliveMother', 'isAliveFather',
                            'isAliveHeir', 'isAliveSpouse','spouse'])
##############################################################################
#Creating dummies
# One-Hot Encoding Qualitative Variables
#title_dummies = pd.get_dummies(list(got1['title']), drop_first = True)
#culture_dummies = pd.get_dummies(list(got1['culture']), drop_first = True)
#house_dummies = pd.get_dummies(list(got1['house']), drop_first = True)



# Concatenating One-Hot Encoded Values with the Larger DataFrame
#got2 = pd.concat(
 #       [got1.loc[:,:],
  #       title_dummies, culture_dummies, house_dummies],
   #      axis = 1)
#got3 = got2.drop(columns=['title', 'culture', 'house'])





# Scalling our Data 

from sklearn.preprocessing import StandardScaler # standard scaler

# Removing the target variable which doesn't need scaling.

got_data = got.drop(['name','isAlive','title','culture','dateOfBirth',
                           'mother','father','heir','house','spouse'],axis=1)

# Instantiating a StandardScaler() object
scaler = StandardScaler()


# Fitting the scaler with our data
scaler.fit(got_data)


# Transforming our data after fit
X_scaled = scaler.transform(got_data)


# Putting our scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)

# Adding labels to our scaled DataFrame
X_scaled_df.columns = got_data.columns


########################
#Train_test_split 
########################
# Importing new libraries
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.model_selection import cross_val_score # k-folds cross validation

got_data   = X_scaled_df

got_target = got.loc[:, 'isAlive']

X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)

####################################### KNN ##################################
# Creating two lists, one for training set accuracy and the other for test
# set accuracy
training_accuracy = []
test_accuracy = []



# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


# Plotting the visualization
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show() 

#print(testaccuracy.index(max(test_accuracy))) -> to index lists
print(test_accuracy.index(max(test_accuracy)))
 
# Building a model with k =16
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 16)

# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)

# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_knn_optimal)

# Generating Predictions based on the optimal KNN model
knn_reg_optimal_pred_train = knn_reg_fit.predict(X_train)
knn_reg_optimal_pred_test = knn_reg_fit.predict(X_test)
#AUC Score 
from sklearn.metrics import roc_auc_score
print('Training AUC Score:',roc_auc_score(
        y_train,knn_reg_optimal_pred_train),round(4))
print('Testing AUC Score:',roc_auc_score(
        y_test,knn_reg_optimal_pred_test),round(4))

# Cross-Validation on knn (cv = 3)

cv_tree_3 = cross_val_score(knn_reg_fit,
                             got_data,
                             got_target,
                             cv = 3)


print(cv_tree_3)


print(pd.np.mean(cv_tree_3).round(3))

###############################################################################
# Logistic Regression 
###############################################################################
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver = 'lbfgs',
                            C = 1)
logreg_fit = logreg.fit(X_train, y_train)

logreg_pred = logreg_fit.predict(X_test)

print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))

#AUC Score 
# Generating Predictions based on the optimal model
logreg_fit_train = logreg_fit.predict(X_train)

logreg_fit_train_test = logreg_fit.predict(X_test)

print('Training AUC Score:',roc_auc_score(
        y_train,logreg_fit_train).round(4))
print('Testing AUC Score:',roc_auc_score(
        y_test,logreg_fit_train_test).round(4))

# Visualizing a confusion matrix

print(confusion_matrix(y_true = y_test,
                       y_pred = logreg_pred))



Predictions = pd.DataFrame({
        'Actual' : y_test,
        'Logreg Prediction' : logreg_pred})
Predictions.to_excel("Model_Predictions.xlsx")
    

import seaborn as sns

labels = ['Alive', 'Dead']

cm = confusion_matrix(y_true = y_test,
                      y_pred = logreg_pred)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'Greys')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()

# Cross-Validation on c_tree_optimal (cv = 3)

cv_tree_3 = cross_val_score(logreg_fit,
                             got_data,
                             got_target,
                             cv = 3)

print(cv_tree_3)

print(pd.np.mean(cv_tree_3).round(3))
###############################################################################
# Decision Tree
###############################################################################
from sklearn.tree import DecisionTreeRegressor # Regression trees

got_data   = got.drop(['isAlive','name','title','culture','dateOfBirth',
                           'mother','father','heir','house','spouse'],axis=1)


got_target = got.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)


# Full tree.
tree_full = DecisionTreeRegressor(random_state = 508)
tree_full.fit(X_train, y_train)

print('Training Score', tree_full.score(X_train, y_train).round(4))
print('Testing Score:', tree_full.score(X_test, y_test).round(4))


########################
# Model adjustments 
########################

tree_leaf_50 = DecisionTreeRegressor(criterion = 'mse',
                                     min_samples_leaf = 50,
                                     random_state = 508)

tree_leaf_50.fit(X_train, y_train)

print('Training Score', tree_leaf_50.score(X_train, y_train).round(4))
print('Testing Score:', tree_leaf_50.score(X_test, y_test).round(4))



# visualizing feature importance

########################
def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')
########################

########################
# Tree with the important features
########################
plot_feature_importances(tree_leaf_50,
                         train = X_train,
                         export = True)

#Dropping the non important features
got_data   = got.drop(['isAlive',
                       'name',
                       'title',
                       'culture',
                       'dateOfBirth',
                        'mother',
                        'father',
                        'heir',
                         'house',
                        'spouse',
                        'm_age',
                        'm_isAliveSpouse',
                        'm_isAliveHeir',
                        'm_isAliveFather',
                        'm_isAliveMother',
                        'm_spouse',
                        'm_heir',
                        'm_mother',
                        'm_father',
                        'm_culture',
                        'numDeadRelations',
                        'age',
                        'book2_A_Clash_Of_Kings',
                        'isNoble',
                        'isMarried',
                        'isAliveSpouse',
                        'isAliveHeir',
                        'isAliveFather',
                        'isAliveMother',
                        ],axis=1)

got_target = got.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)

# Cross-Validation on tree_leaf_50 (cv = 3)

cv_tree_3 = cross_val_score(tree_leaf_50,
                             got_data,
                             got_target,
                             cv = 3)


print(cv_tree_3)


print(pd.np.mean(cv_tree_3).round(3))

###############################################################################
# Random Forest
###############################################################################
# Loading new libraries
from sklearn.ensemble import RandomForestClassifier

got_data   = got.drop(['isAlive',
                       'name',
                       'title',
                       'culture',
                       'dateOfBirth',
                        'mother',
                        'father',
                        'heir',
                         'house',
                        'spouse',
                        'm_age',
                        'm_isAliveSpouse',
                        'm_isAliveHeir',
                        'm_isAliveFather',
                        'm_isAliveMother',
                        'm_spouse',
                        'm_heir',
                        'm_mother',
                        'm_father',
                        'm_culture',
                        'numDeadRelations',
                        'age',
                        'book2_A_Clash_Of_Kings',
                        'isNoble',
                        'isMarried',
                        'isAliveSpouse',
                        'isAliveHeir',
                        'isAliveFather',
                        'isAliveMother',
                        ],axis=1)


got_target = got.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)


# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)


full_entropy_fit = full_forest_entropy.fit(X_train, y_train)


# Are our predictions the same for each model? 
pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))


full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()



# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))


# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))


# Saving score objects
gini_full_train = full_gini_fit.score(X_train, y_train)
gini_full_test  = full_gini_fit.score(X_test, y_test)

entropy_full_train = full_entropy_fit.score(X_train, y_train)
entropy_full_test  = full_entropy_fit.score(X_test, y_test)








#My gridsearch was taking a long time (over 8 hours), I am not sure if it's my 
#laptop that is an older model with a lower processing power. I was forced to drop 
#it and relied on the results that I found through the other models

