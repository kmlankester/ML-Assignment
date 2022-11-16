'''

Example of how to run the code with the COVID19 dataset:

covid_dataset = read_dataset('latestdata.csv')
covid_dataset = clean(covid_dataset)
models(covid_dataset)

'''

import datetime
import pandas as pd
import pycountry_convert as pc
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix


# read the dataset into a dataframe from a scv file.
def read_dataset(filename):
    # low memory is set to false due to the size of the csv file
    dataset = pd.read_csv(filename, low_memory = False)
    return dataset


# return an integer age value from a list of age strings
def get_age(age):
    values = [float(i) for i in age if len(i) != 0]
    return sum(values)/len(values)


# return the continent corresponding to a given country
def get_continent(country):
    alpha2 = pc.country_name_to_country_alpha2(country)
    cont_code = pc.country_alpha2_to_continent_code(alpha2)
    cont_name = pc.convert_continent_code_to_continent_name(cont_code)
    return cont_name


# calculate a date object from a given date string
# return the difference in days from a date object corresponding to 01/01/2020
def get_date(unformatted_date):
    if type(unformatted_date) == float:
        return
    start_date = datetime.datetime.strptime('01/01/2020', '%d/%m/%Y').date()
    date_string = unformatted_date[:10]
    date_string = date_string.replace('.', '/')
    date_format = '%d/%m/%Y'
    output_date = datetime.datetime.strptime(date_string, date_format).date()
    delta = (output_date - start_date).days
    return delta


# return the variance inflation factor of a given set of features
def vif(dataset):
    vif = pd.DataFrame()
    vif['feature'] = dataset.columns
    vif['VIF'] = [variance_inflation_factor(dataset.values, i) for i in range(dataset.shape[1])]
    return vif


# return a dataset where missing values have iteratively been imputed
def impute_data(dataset):
    impute = IterativeImputer(max_iter=10, verbose=0)
    impute.fit(dataset)
    imputed_dataset = impute.transform(dataset)
    imputed_dataset = pd.DataFrame(imputed_dataset, columns = dataset.columns)
    return imputed_dataset


# returns a cleaned dataset with the chosen features in the project
# the returned dataset is complete with no missing values
def clean(dataset):

    # remove unwanted features from the dataset
    dataset = dataset.loc[:, ['age',
                              'sex',
                              'country',
                              'chronic_disease_binary',
                              'travel_history_binary',
                              'date_confirmation',
                              'date_onset_symptoms',
                              'outcome']]

    # drop the samples that have missing values relating to the specified features
    # reset the index once the samples have been removed
    dataset.dropna(subset = ['age', 'sex', 'country', 'chronic_disease_binary', 
                             'travel_history_binary', 'date_confirmation', 'outcome'], 
                   inplace = True)
    dataset = dataset.reset_index(drop = True)

    # convert date strings in the dataset to integer representations
    # the integer represents the difference of days between the date and 01/01/2020
    dataset['date_confirmation'] = dataset['date_confirmation'].apply(get_date)
    dataset['date_onset_symptoms'] = dataset['date_onset_symptoms'].apply(get_date)

    # convert age strings to integer values
    dataset['age'] = dataset['age'].str.split('-')
    dataset['age'] = dataset['age'].apply(get_age)

    # convert sex to integer values corresponding to male and female
    dataset.loc[dataset['sex'].str.lower() == 'male', 'sex'] = 0
    dataset.loc[dataset['sex'].str.lower() == 'female', 'sex'] = 1
    dataset['sex'] = pd.to_numeric(dataset['sex'])

    # convert chronic disease boolean values to integer values corresponding to true and false
    dataset.loc[dataset['chronic_disease_binary'] == True, 'chronic_disease_binary'] = 1
    dataset.loc[dataset['chronic_disease_binary'] == False, 'chronic_disease_binary'] = 0
    dataset['chronic_disease_binary'] = pd.to_numeric(dataset['chronic_disease_binary'])

    # convert travel history boolean values to integer values corresponding to true and false
    dataset.loc[dataset['travel_history_binary'] == True, 'travel_history_binary'] = 1
    dataset.loc[dataset['travel_history_binary'] == False, 'travel_history_binary'] = 0
    dataset['travel_history_binary'] = pd.to_numeric(dataset['travel_history_binary'])

    # create the positive and negative outcomes to be used as the label
    # outcomes are categorised according to the severity of the case
    outcomes = ['alive', 'recover', 'stable', 'discharge', 'released', 'treated', 'not hospitalized']
    outcome_pattern = '|'.join(outcomes)
    dataset.loc[dataset['outcome'].str.lower().str.contains(outcome_pattern) == False, 'outcome'] = 1
    dataset.loc[dataset['outcome'].str.lower().str.contains(outcome_pattern) == True, 'outcome'] = 0
    dataset['outcome'] = dataset['outcome'].astype('float64')

    # encode the countries feature using one-hot encoding  
    '''  
    dataset['country'] = dataset['country'].apply(get_continent)
    dataset.rename(columns={'country': 'continent'}, inplace = True)
    country_encoder = OneHotEncoder(sparse = False)
    temp_country = pd.DataFrame(country_encoder.fit_transform(dataset[['country']]))
    temp_country.columns = country_encoder.get_feature_names(['country'])
    dataset.drop(['country'], axis = 1, inplace = True)
    dataset = pd.concat([temp_country, dataset], axis = 1)
    '''

    # calculate variance inflation factor values
    '''
    temp_vif = dataset.iloc[:,:-3]
    example_vif = vif(temp_vif)
    example_vif.to_csv('vif_scores.csv', index = False)
    '''

    # iteratively impute missing values belonging to the date_onset_symptoms feature
    # create a new feature combining date_confirmation and date_onset_symptoms, remove the latter two features
    temp_impute = dataset.iloc[:,:-1]
    temp_impute = impute_data(temp_impute)
    temp_impute['symptoms_to_conf'] = temp_impute['date_confirmation'] - temp_impute['date_onset_symptoms']
    temp_impute.drop(['date_confirmation', 'date_onset_symptoms'], axis = 1, inplace = True)
    temp_impute.loc[temp_impute['symptoms_to_conf'] > 21.0, 'symptoms_to_conf'] = 21.0
    temp_impute.loc[temp_impute['symptoms_to_conf'] < -21.0, 'symptoms_to_conf'] = -21.0

    # recombine the outcome label with the imputed feature dataset
    dataset = pd.concat([temp_impute, dataset.iloc[:,-1:]], axis = 1)

    # plot the distribution of symptoms_to_conf feature values
    '''
    dataset['symptoms_to_conf'].hist(bins=40)
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.show()
    '''
    
    return dataset


# train a random forest classifier, k neighbors classifier, and support vector classifier on a given dataset
def models(dataset):

    # split the dataset into training and testings sets with a ratio of 75:25
    X, y = dataset.drop(dataset.iloc[:,-1:], axis = 1), dataset['outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 10)

    # standardization of the training set
    '''
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    '''

    # normalization of the training set
    '''
    mmscaler = MinMaxScaler()
    X_train = mmscaler.fit_transform(X_train)
    X_test = mmscaler.transform(X_test)
    '''
    
    # kfold = StratifiedKFold(n_splits = 5)

    # build random forest classifier using selected hyperparameters
    rf_classifier = RandomForestClassifier(n_estimators = 200, max_features = 'auto',
                                           max_depth = 8, criterion = 'gini', random_state = 10)
    
    # results_kfold = cross_val_score(rf_classifier, X, y, cv=kfold)
    # print('Accuracy: ' + str(results_kfold.mean()*100))

    # build default random forest classifier
    # rf_classifier = RandomForestClassifier()

    # train the random forest model
    # calculate the predicted outcomes, probabilities, and roc_value
    rf_classifier.fit(X_train, y_train)
    rf_predicted = rf_classifier.predict(X_test)
    rf_probability = rf_classifier.predict_proba(X)[:,1]
    rf_roc_value = roc_auc_score(y, rf_probability)
    
    print('RF')
    print('Accuracy: ' + str(accuracy_score(y_test, rf_predicted)))
    print('Precision: ' + str(precision_score(y_test, rf_predicted, average='macro')))
    print('Recall: ' + str(recall_score(y_test, rf_predicted, average='macro')))
    print('ROC_AUC: ' + str(rf_roc_value))

    # output the confusion matrix to a png file
    '''
    rf_confusion_matrix = pd.crosstab(y_test, rf_predicted, rownames = ['Actual'], colnames = ['Predicted'])
    rf_conf = sns.heatmap(rf_confusion_matrix, annot = True, fmt='d')
    rf_conf.figure.savefig('rf_conf_default.png')
    '''
    
    # kfold = StratifiedKFold(n_splits = 5)

    # build k neighbors classifier using selected hyperparameters
    kn_classifier = KNeighborsClassifier(n_neighbors = 8, weights = 'uniform')
    
    # results_kfold = cross_val_score(kn_classifier, X, y, cv=kfold)
    # print('Accuracy: ' + str(results_kfold.mean()*100))

    # build default k neighbors classifier
    # kn_classifier = KNeighborsClassifier()

    # train the k neighbors model
    # calculate the predicted outcomes, probabilities, and roc_value
    kn_classifier.fit(X_train, y_train)
    kn_predicted = kn_classifier.predict(X_test)
    kn_probability = kn_classifier.predict_proba(X)[:,1]
    kn_roc_value = roc_auc_score(y, kn_probability)
    
    print('KN')
    print('Accuracy: ' + str(accuracy_score(y_test, kn_predicted)))
    print('Precision: ' + str(precision_score(y_test, kn_predicted, average='macro')))
    print('Recall: ' + str(recall_score(y_test, kn_predicted, average='macro')))
    print('ROC_AUC: ' + str(kn_roc_value))

    # output the confusion matrix to a png file
    '''   
    kn_confusion_matrix = pd.crosstab(y_test, kn_predicted, rownames = ['Actual'], colnames = ['Predicted'])
    kn_conf = sns.heatmap(kn_confusion_matrix, annot = True, fmt='d')
    kn_conf.figure.savefig('kn_conf_default.png')
    '''

    # kfold = StratifiedKFold(n_splits = 5)

    # build support vector classifier using selected hyperparameters
    svc_classifier = SVC(C=1000, gamma=0.0001, kernel='rbf', probability=True)
    
    # results_kfold = cross_val_score(svc_classifier, X, y, cv=kfold)
    # print('Accuracy: ' + str(results_kfold.mean()*100))    

    # build default support vector classifier
    # svc_classifier = SVC(probability=True)

    # train the support vector model
    # calculate the predicted outcomes, probabilities, and roc_value
    svc_classifier.fit(X_train, y_train)
    svc_predicted = svc_classifier.predict(X_test)
    svc_probability = svc_classifier.predict_proba(X)[:,1]
    svc_roc_value = roc_auc_score(y, svc_probability)
    
    print('SVC')
    print('Accuracy: ' + str(accuracy_score(y_test, svc_predicted)))
    print('Precision: ' + str(precision_score(y_test, svc_predicted, average='macro')))
    print('Recall: ' + str(recall_score(y_test, svc_predicted, average='macro')))
    print('ROC_AUC: ' + str(kn_roc_value))

    # output the confusion matrix to a png file 
    '''  
    svc_confusion_matrix = pd.crosstab(y_test, svc_predicted, rownames = ['Actual'], colnames = ['Predicted'])
    svc_conf = sns.heatmap(svc_confusion_matrix, annot = True, fmt='d')
    svc_conf.figure.savefig('svc_conf_default.png')
    '''
