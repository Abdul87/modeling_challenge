import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

# define variables
no_folds = 5 # Number of folds for the cross validation

# Load the data and add the column names
col_names = pd.read_csv("header.txt", sep=',', header=None)
all_data = pd.read_csv('data.csv')
all_data.columns = col_names[0]

## ------ Data cleaning ------ ##
# Find the nans in each row, if row has nan then remove that row
all_data.isnull().sum()
cleaned_data = all_data.dropna(axis=0)
cleaned_data.isnull().sum()

# Encode the text data
labels_state, uniques_state = pd.factorize(cleaned_data['client_state'])
labels_BC, uniques_BC = pd.factorize(cleaned_data['BC'])

# insert the new columns in the cleaned_data
cleaned_data['client_state'] = labels_state
cleaned_data['BC'] = labels_BC

# Extract the ylabels and put them into new Dataframe
y_cleaned_data = cleaned_data['churn']
x_cleaned_data = cleaned_data.drop(columns=['churn'])

# Feature selection
#x_cleaned_data = cleaned_data[['CPL_wrt_BC', 'duration', 'avg_budget', 'clicks']]

# Normalize the data if needed
x_cleaned_data = (x_cleaned_data - x_cleaned_data.mean()) / (x_cleaned_data.std())

### ------ Train and test the data ------ ###

# Decision Trees
print('Decision Trees')
DT_model = DecisionTreeClassifier(max_depth=5)
cv_DT = cross_validate(DT_model, x_cleaned_data, y_cleaned_data, cv=no_folds, scoring=('accuracy', 'precision', 'recall'))
print(cv_DT['test_accuracy'].mean())
print(cv_DT['test_precision'].mean())
print(cv_DT['test_recall'].mean())

# kNN
print('------------')
print('Knn')
kNN_model = KNeighborsClassifier(n_neighbors=8, weights='distance')
cv_kNN = cross_validate(kNN_model, x_cleaned_data, y_cleaned_data, cv=no_folds, scoring=('accuracy', 'precision',  'recall'))
print(cv_kNN['test_accuracy'].mean())
print(cv_kNN['test_precision'].mean())
print(cv_kNN['test_recall'].mean())

# SVM
print('------------')
print('SVM')
SVM_model =  SVC(kernel='rbf', C=1e3, gamma=0.1)
cv_SVM = cross_validate(SVM_model, x_cleaned_data, y_cleaned_data, cv=no_folds, scoring=('accuracy', 'precision', 'recall'))
print(cv_SVM['test_accuracy'].mean())
print(cv_SVM['test_precision'].mean())
print(cv_SVM['test_recall'].mean())

# Neural Networks
print('------------')
print('Deep Learning')
NN_model = MLPClassifier(hidden_layer_sizes=(15, 5,), activation='relu', solver='adam', alpha=0.00001,
                                batch_size=1, learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5,
                                max_iter=1000,
                                shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,
                                momentum=0.9,
                                nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                                beta_2=0.999,
                                epsilon=1e-08)
cv_NN = cross_validate(NN_model, x_cleaned_data, y_cleaned_data, cv=no_folds, scoring=('accuracy', 'precision', 'recall'))
print(cv_NN['test_accuracy'].mean())
print(cv_NN['test_precision'].mean())
print(cv_NN['test_recall'].mean())