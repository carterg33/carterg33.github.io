---
title: "From Scratch: Can Neural Networks Land a Spaceship?"
excerpt: "A simple perceptron model coded completely from scratch without the use of Python libraries - capable of learning to land a spaceship (kind of).<br/><img src='/images/500x300.png'>"
collection: portfolio
---
<font size="20">
# Classification Model

### Import Libraries & Load Data


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df = pd.read_csv('CE802_P2_Data.csv')
```

## Data Summary Statistics


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>F2</th>
      <th>F3</th>
      <th>F4</th>
      <th>F5</th>
      <th>F6</th>
      <th>F7</th>
      <th>F8</th>
      <th>F9</th>
      <th>F10</th>
      <th>F11</th>
      <th>F12</th>
      <th>F13</th>
      <th>F14</th>
      <th>F15</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16</td>
      <td>6.03</td>
      <td>-4.56</td>
      <td>30.00</td>
      <td>63.04</td>
      <td>0.55</td>
      <td>-0.78</td>
      <td>0.96</td>
      <td>-1.67</td>
      <td>1</td>
      <td>-339.75</td>
      <td>0.84</td>
      <td>9.90</td>
      <td>-2.22</td>
      <td>-9.53</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>1.41</td>
      <td>-4.24</td>
      <td>31.38</td>
      <td>100.04</td>
      <td>0.06</td>
      <td>1.08</td>
      <td>0.60</td>
      <td>-3.67</td>
      <td>1</td>
      <td>-246.75</td>
      <td>-0.62</td>
      <td>12.32</td>
      <td>-0.56</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>630</td>
      <td>43.35</td>
      <td>0.50</td>
      <td>50.70</td>
      <td>219.04</td>
      <td>-3.46</td>
      <td>10.08</td>
      <td>5.65</td>
      <td>-14.67</td>
      <td>10</td>
      <td>-129.75</td>
      <td>-3.40</td>
      <td>1.80</td>
      <td>8.48</td>
      <td>-10.29</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>2.79</td>
      <td>-3.58</td>
      <td>31.83</td>
      <td>69.04</td>
      <td>-0.40</td>
      <td>-0.09</td>
      <td>0.18</td>
      <td>-1.67</td>
      <td>1</td>
      <td>-276.75</td>
      <td>-0.32</td>
      <td>13.32</td>
      <td>1.36</td>
      <td>-9.92</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18</td>
      <td>0.36</td>
      <td>-6.62</td>
      <td>32.97</td>
      <td>75.04</td>
      <td>2.43</td>
      <td>0.24</td>
      <td>0.48</td>
      <td>-0.67</td>
      <td>1</td>
      <td>-294.75</td>
      <td>3.14</td>
      <td>10.80</td>
      <td>-5.86</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1500 entries, 0 to 1499
    Data columns (total 16 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   F1      1500 non-null   int64  
     1   F2      1500 non-null   float64
     2   F3      1500 non-null   float64
     3   F4      1500 non-null   float64
     4   F5      1500 non-null   float64
     5   F6      1500 non-null   float64
     6   F7      1500 non-null   float64
     7   F8      1500 non-null   float64
     8   F9      1500 non-null   float64
     9   F10     1500 non-null   int64  
     10  F11     1500 non-null   float64
     11  F12     1500 non-null   float64
     12  F13     1500 non-null   float64
     13  F14     1500 non-null   float64
     14  F15     750 non-null    float64
     15  Class   1500 non-null   bool   
    dtypes: bool(1), float64(13), int64(2)
    memory usage: 177.4 KB
    


*   All features are either 'float' or 'int'
*   F15 contains 50% missing values




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>F2</th>
      <th>F3</th>
      <th>F4</th>
      <th>F5</th>
      <th>F6</th>
      <th>F7</th>
      <th>F8</th>
      <th>F9</th>
      <th>F10</th>
      <th>F11</th>
      <th>F12</th>
      <th>F13</th>
      <th>F14</th>
      <th>F15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1500.000000</td>
      <td>1500.00000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>1500.000000</td>
      <td>750.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>218.049333</td>
      <td>14.90628</td>
      <td>-0.829733</td>
      <td>41.063440</td>
      <td>133.485333</td>
      <td>-1.592947</td>
      <td>5.390220</td>
      <td>3.854280</td>
      <td>-7.308000</td>
      <td>5.403333</td>
      <td>-307.992000</td>
      <td>0.244707</td>
      <td>7.315067</td>
      <td>3.542427</td>
      <td>-10.463013</td>
    </tr>
    <tr>
      <th>std</th>
      <td>234.669160</td>
      <td>12.97563</td>
      <td>5.090345</td>
      <td>9.500727</td>
      <td>71.416874</td>
      <td>2.513850</td>
      <td>7.662813</td>
      <td>3.465276</td>
      <td>7.159315</td>
      <td>4.504907</td>
      <td>120.565344</td>
      <td>2.217929</td>
      <td>4.982472</td>
      <td>5.081288</td>
      <td>1.005117</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>-17.560000</td>
      <td>30.000000</td>
      <td>54.040000</td>
      <td>-6.070000</td>
      <td>-19.800000</td>
      <td>0.000000</td>
      <td>-24.670000</td>
      <td>0.000000</td>
      <td>-609.750000</td>
      <td>-4.480000</td>
      <td>-1.700000</td>
      <td>-14.520000</td>
      <td>-12.820000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.000000</td>
      <td>3.72000</td>
      <td>-4.600000</td>
      <td>32.160000</td>
      <td>78.040000</td>
      <td>-4.000000</td>
      <td>-0.300000</td>
      <td>0.600000</td>
      <td>-14.670000</td>
      <td>1.000000</td>
      <td>-354.750000</td>
      <td>-1.060000</td>
      <td>2.560000</td>
      <td>-0.220000</td>
      <td>-11.210000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>100.000000</td>
      <td>6.21000</td>
      <td>-2.810000</td>
      <td>33.240000</td>
      <td>98.040000</td>
      <td>-0.610000</td>
      <td>2.400000</td>
      <td>1.020000</td>
      <td>-3.670000</td>
      <td>1.000000</td>
      <td>-291.750000</td>
      <td>-0.060000</td>
      <td>9.220000</td>
      <td>1.580000</td>
      <td>-10.470000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>390.000000</td>
      <td>24.60000</td>
      <td>4.080000</td>
      <td>49.350000</td>
      <td>194.040000</td>
      <td>0.350000</td>
      <td>12.877500</td>
      <td>7.050000</td>
      <td>-1.670000</td>
      <td>10.000000</td>
      <td>-252.750000</td>
      <td>1.400000</td>
      <td>10.985000</td>
      <td>8.500000</td>
      <td>-9.742500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>780.000000</td>
      <td>47.10000</td>
      <td>8.300000</td>
      <td>60.450000</td>
      <td>299.040000</td>
      <td>6.070000</td>
      <td>18.810000</td>
      <td>10.350000</td>
      <td>1.330000</td>
      <td>10.000000</td>
      <td>35.250000</td>
      <td>8.720000</td>
      <td>22.780000</td>
      <td>12.860000</td>
      <td>-7.740000</td>
    </tr>
  </tbody>
</table>
</div>



*   Data has very different ranges so will need to be scaled




```python
sns.countplot(x="Class", data=df)
```




    <AxesSubplot:xlabel='Class', ylabel='count'>




    
![png](classification_model_files/classification_model_10_1.png)
    


## Data Modelling


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

```


```python
X = df.drop(['Class'], axis=1)
y = df['Class']
```


```python
# initiate scaler
scaler = StandardScaler()

# create dictionary of imputation methods
imputation_methods = {
                      "const": SimpleImputer(strategy='constant', fill_value = 0, missing_values=np.nan), 
                      "mean": SimpleImputer(strategy='mean', missing_values=np.nan), 
                      "knn": KNNImputer(missing_values=np.nan),
                      "iter": IterativeImputer(missing_values=np.nan, random_state = 0) 
}

# create cross validation splits using KFold
inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)

```

### Create Parameter Space


```python
# create paramater space per model
param_grid_tree = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_features' : [None, 'auto', 'sqrt', 'log2']
}

param_grid_forest = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_features' : ['auto', 'sqrt', 'log2']
}

param_grid_reg = {
    'classifier__solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

param_grid_knn = {
    'classifier__n_neighbors': [1,3,5,7,10,15,20],
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
    'classifier__weights' : ['uniform', 'distance']
}

param_grid_svm = {
    'classifier__kernel': ['poly', 'rbf', 'sigmoid'],
    'classifier__C': [0.01, 1, 5, 10, 100]
}

# create dictionary of classifiers with the classifier and paramaters as value 
classifiers = {
              "tree": (DecisionTreeClassifier(), param_grid_tree),
              "forest": (RandomForestClassifier(), param_grid_forest),
               "reg": (LogisticRegression(), param_grid_reg),
               'knn': (KNeighborsClassifier(), param_grid_knn),
               'svm': (SVC(), param_grid_svm)
}

```


```python
# empty dictionary to store each model pipeline and respective params as values
models_params = {}

# loop over to fill dictionary
for classifier_name, classifier in classifiers.items():
  for method_name, method in imputation_methods.items():
    pipe = Pipeline(steps=[('imputer', method),
                           ('scaler', scaler),
                           ('classifier', classifier[0])])
    models_params[f"{classifier_name}_{method_name}"] = (pipe, classifier[1])
```


```python
from sklearn.metrics import classification_report, accuracy_score, make_scorer

# variables initiated for classification report
originalclass = []
predictedclass = []

# define custom scoring function
def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred) # return accuracy score
```

### Finding the Best Model


```python
avg_outer_score = dict()

for name, (pipe, params) in models_params.items():
    originalclass = []
    predictedclass = []
    # compute nested cross validation using GridSearchCV to find the optimal model for that pipeline
    optimised_model = GridSearchCV(pipe, params, cv=inner_cv, scoring='accuracy')
    scores = cross_val_score(optimised_model, X, y, cv=outer_cv, scoring=make_scorer(classification_report_with_accuracy_score))
    # get the average of the outer fold scores
    avg_outer_score[name] = np.mean(scores)
    rounded_scores = [f"{score:.2f}" for score in scores] #rounded scores for print statement
    print(f"{name}\nAccuracy in the outer folds: {rounded_scores}.\nAverage Error: {np.mean(scores):.2f}")
    print()
    print(classification_report(originalclass, predictedclass)) 

```

    tree_const
    Accuracy in the outer folds: ['0.81', '0.77', '0.79', '0.77', '0.81'].
    Average Error: 0.79
    
                  precision    recall  f1-score   support
    
           False       0.80      0.82      0.81       809
            True       0.78      0.76      0.77       691
    
        accuracy                           0.79      1500
       macro avg       0.79      0.79      0.79      1500
    weighted avg       0.79      0.79      0.79      1500
    
    tree_mean
    Accuracy in the outer folds: ['0.79', '0.80', '0.82', '0.77', '0.78'].
    Average Error: 0.79
    
                  precision    recall  f1-score   support
    
           False       0.80      0.81      0.81       809
            True       0.78      0.77      0.77       691
    
        accuracy                           0.79      1500
       macro avg       0.79      0.79      0.79      1500
    weighted avg       0.79      0.79      0.79      1500
    
    tree_knn
    Accuracy in the outer folds: ['0.75', '0.76', '0.75', '0.78', '0.76'].
    Average Error: 0.76
    
                  precision    recall  f1-score   support
    
           False       0.77      0.78      0.78       809
            True       0.74      0.73      0.74       691
    
        accuracy                           0.76      1500
       macro avg       0.76      0.76      0.76      1500
    weighted avg       0.76      0.76      0.76      1500
    
    tree_iter
    Accuracy in the outer folds: ['0.81', '0.79', '0.82', '0.77', '0.79'].
    Average Error: 0.79
    
                  precision    recall  f1-score   support
    
           False       0.81      0.80      0.81       809
            True       0.77      0.79      0.78       691
    
        accuracy                           0.79      1500
       macro avg       0.79      0.79      0.79      1500
    weighted avg       0.79      0.79      0.79      1500
    
    forest_const
    Accuracy in the outer folds: ['0.88', '0.83', '0.88', '0.86', '0.84'].
    Average Error: 0.86
    
                  precision    recall  f1-score   support
    
           False       0.86      0.89      0.87       809
            True       0.86      0.82      0.84       691
    
        accuracy                           0.86      1500
       macro avg       0.86      0.86      0.86      1500
    weighted avg       0.86      0.86      0.86      1500
    
    forest_mean
    Accuracy in the outer folds: ['0.90', '0.84', '0.88', '0.87', '0.87'].
    Average Error: 0.87
    
                  precision    recall  f1-score   support
    
           False       0.88      0.89      0.88       809
            True       0.87      0.86      0.86       691
    
        accuracy                           0.87      1500
       macro avg       0.87      0.87      0.87      1500
    weighted avg       0.87      0.87      0.87      1500
    
    forest_knn
    Accuracy in the outer folds: ['0.89', '0.82', '0.87', '0.87', '0.88'].
    Average Error: 0.86
    
                  precision    recall  f1-score   support
    
           False       0.86      0.89      0.88       809
            True       0.87      0.83      0.85       691
    
        accuracy                           0.86      1500
       macro avg       0.86      0.86      0.86      1500
    weighted avg       0.86      0.86      0.86      1500
    
    forest_iter
    Accuracy in the outer folds: ['0.88', '0.86', '0.90', '0.91', '0.87'].
    Average Error: 0.88
    
                  precision    recall  f1-score   support
    
           False       0.89      0.89      0.89       809
            True       0.87      0.87      0.87       691
    
        accuracy                           0.88      1500
       macro avg       0.88      0.88      0.88      1500
    weighted avg       0.88      0.88      0.88      1500
    
    reg_const
    Accuracy in the outer folds: ['0.83', '0.80', '0.83', '0.83', '0.76'].
    Average Error: 0.81
    
                  precision    recall  f1-score   support
    
           False       0.84      0.80      0.82       809
            True       0.78      0.82      0.80       691
    
        accuracy                           0.81      1500
       macro avg       0.81      0.81      0.81      1500
    weighted avg       0.81      0.81      0.81      1500
    
    reg_mean
    Accuracy in the outer folds: ['0.88', '0.83', '0.84', '0.83', '0.84'].
    Average Error: 0.84
    
                  precision    recall  f1-score   support
    
           False       0.85      0.87      0.86       809
            True       0.84      0.82      0.83       691
    
        accuracy                           0.84      1500
       macro avg       0.84      0.84      0.84      1500
    weighted avg       0.84      0.84      0.84      1500
    
    reg_knn
    Accuracy in the outer folds: ['0.84', '0.82', '0.85', '0.80', '0.81'].
    Average Error: 0.82
    
                  precision    recall  f1-score   support
    
           False       0.84      0.83      0.84       809
            True       0.81      0.81      0.81       691
    
        accuracy                           0.82      1500
       macro avg       0.82      0.82      0.82      1500
    weighted avg       0.82      0.82      0.82      1500
    
    

    C:\Users\carte\miniconda3\envs\ce889\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    C:\Users\carte\miniconda3\envs\ce889\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    C:\Users\carte\miniconda3\envs\ce889\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    C:\Users\carte\miniconda3\envs\ce889\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    C:\Users\carte\miniconda3\envs\ce889\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    C:\Users\carte\miniconda3\envs\ce889\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    C:\Users\carte\miniconda3\envs\ce889\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    C:\Users\carte\miniconda3\envs\ce889\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    C:\Users\carte\miniconda3\envs\ce889\lib\site-packages\sklearn\linear_model\_sag.py:329: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      warnings.warn("The max_iter was reached which means "
    

    reg_iter
    Accuracy in the outer folds: ['0.89', '0.84', '0.86', '0.85', '0.85'].
    Average Error: 0.86
    
                  precision    recall  f1-score   support
    
           False       0.88      0.86      0.87       809
            True       0.84      0.86      0.85       691
    
        accuracy                           0.86      1500
       macro avg       0.86      0.86      0.86      1500
    weighted avg       0.86      0.86      0.86      1500
    
    knn_const
    Accuracy in the outer folds: ['0.75', '0.75', '0.78', '0.75', '0.79'].
    Average Error: 0.76
    
                  precision    recall  f1-score   support
    
           False       0.76      0.83      0.79       809
            True       0.78      0.68      0.73       691
    
        accuracy                           0.76      1500
       macro avg       0.77      0.76      0.76      1500
    weighted avg       0.76      0.76      0.76      1500
    
    knn_mean
    Accuracy in the outer folds: ['0.78', '0.80', '0.81', '0.81', '0.81'].
    Average Error: 0.80
    
                  precision    recall  f1-score   support
    
           False       0.78      0.87      0.82       809
            True       0.82      0.72      0.77       691
    
        accuracy                           0.80      1500
       macro avg       0.80      0.79      0.80      1500
    weighted avg       0.80      0.80      0.80      1500
    
    knn_knn
    Accuracy in the outer folds: ['0.79', '0.78', '0.83', '0.79', '0.81'].
    Average Error: 0.80
    
                  precision    recall  f1-score   support
    
           False       0.79      0.86      0.82       809
            True       0.82      0.73      0.77       691
    
        accuracy                           0.80      1500
       macro avg       0.80      0.79      0.80      1500
    weighted avg       0.80      0.80      0.80      1500
    
    knn_iter
    Accuracy in the outer folds: ['0.87', '0.82', '0.86', '0.88', '0.87'].
    Average Error: 0.86
    
                  precision    recall  f1-score   support
    
           False       0.84      0.90      0.87       809
            True       0.88      0.81      0.84       691
    
        accuracy                           0.86      1500
       macro avg       0.86      0.85      0.86      1500
    weighted avg       0.86      0.86      0.86      1500
    
    svm_const
    Accuracy in the outer folds: ['0.89', '0.86', '0.89', '0.88', '0.86'].
    Average Error: 0.88
    
                  precision    recall  f1-score   support
    
           False       0.89      0.88      0.89       809
            True       0.86      0.87      0.87       691
    
        accuracy                           0.88      1500
       macro avg       0.88      0.88      0.88      1500
    weighted avg       0.88      0.88      0.88      1500
    
    svm_mean
    Accuracy in the outer folds: ['0.90', '0.86', '0.89', '0.88', '0.88'].
    Average Error: 0.88
    
                  precision    recall  f1-score   support
    
           False       0.90      0.89      0.89       809
            True       0.87      0.88      0.87       691
    
        accuracy                           0.88      1500
       macro avg       0.88      0.88      0.88      1500
    weighted avg       0.88      0.88      0.88      1500
    
    svm_knn
    Accuracy in the outer folds: ['0.90', '0.86', '0.89', '0.87', '0.90'].
    Average Error: 0.88
    
                  precision    recall  f1-score   support
    
           False       0.89      0.89      0.89       809
            True       0.88      0.87      0.87       691
    
        accuracy                           0.88      1500
       macro avg       0.88      0.88      0.88      1500
    weighted avg       0.88      0.88      0.88      1500
    
    svm_iter
    Accuracy in the outer folds: ['0.91', '0.87', '0.89', '0.90', '0.88'].
    Average Error: 0.89
    
                  precision    recall  f1-score   support
    
           False       0.90      0.89      0.90       809
            True       0.87      0.89      0.88       691
    
        accuracy                           0.89      1500
       macro avg       0.89      0.89      0.89      1500
    weighted avg       0.89      0.89      0.89      1500
    
    


```python
# store best model, associated parameters and score
best_model, best_model_score = max(avg_outer_score.items(),key=(lambda name_averagescore: name_averagescore[1]))
best_model, best_model_params = models_params[name]

print(f"The best model is:\n{best_model} \nWith an average score of: {best_model_score}")
```

    The best model is:
    Pipeline(steps=[('imputer', IterativeImputer(random_state=0)),
                    ('scaler', StandardScaler()), ('classifier', SVC())]) 
    With an average score of: 0.8886666666666667
    

## Applying to the Test Set


```python
test_df = pd.read_csv("CE802_P2_Test.csv")
```


```python
test_inputs_df = test_df.drop(['Class'], axis = 1)
```


```python
# fit our best model to the entire data set
best_model.fit(X,y)
y_pred = best_model.predict(test_inputs_df)
```


```python
# check value counts to see distribution
y_pred = pd.Series(y_pred)
y_pred.value_counts() 
```




    False    802
    True     698
    dtype: int64




```python
# add our predictions back to test dataframe
test_df['Class'] = y_pred
test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1</th>
      <th>F2</th>
      <th>F3</th>
      <th>F4</th>
      <th>F5</th>
      <th>F6</th>
      <th>F7</th>
      <th>F8</th>
      <th>F9</th>
      <th>F10</th>
      <th>F11</th>
      <th>F12</th>
      <th>F13</th>
      <th>F14</th>
      <th>F15</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>5.58</td>
      <td>-4.66</td>
      <td>31.83</td>
      <td>69.04</td>
      <td>-0.35</td>
      <td>-1.29</td>
      <td>0.06</td>
      <td>-3.67</td>
      <td>1</td>
      <td>-243.75</td>
      <td>0.94</td>
      <td>13.84</td>
      <td>-1.48</td>
      <td>-11.04</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80</td>
      <td>37.95</td>
      <td>4.40</td>
      <td>50.70</td>
      <td>199.04</td>
      <td>-4.83</td>
      <td>5.19</td>
      <td>7.25</td>
      <td>-4.67</td>
      <td>10</td>
      <td>-474.75</td>
      <td>-3.34</td>
      <td>0.46</td>
      <td>8.72</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>1.08</td>
      <td>-4.14</td>
      <td>32.13</td>
      <td>73.04</td>
      <td>0.14</td>
      <td>2.01</td>
      <td>0.59</td>
      <td>-3.67</td>
      <td>1</td>
      <td>-234.75</td>
      <td>-1.08</td>
      <td>9.36</td>
      <td>-1.20</td>
      <td>-11.71</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>240</td>
      <td>34.95</td>
      <td>3.74</td>
      <td>44.85</td>
      <td>264.04</td>
      <td>-2.92</td>
      <td>11.52</td>
      <td>8.45</td>
      <td>-14.67</td>
      <td>10</td>
      <td>-174.75</td>
      <td>-3.20</td>
      <td>2.94</td>
      <td>4.14</td>
      <td>-10.40</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>4.11</td>
      <td>-3.78</td>
      <td>31.92</td>
      <td>92.04</td>
      <td>1.09</td>
      <td>-2.67</td>
      <td>0.72</td>
      <td>-3.67</td>
      <td>1</td>
      <td>-282.75</td>
      <td>-0.40</td>
      <td>11.20</td>
      <td>0.92</td>
      <td>-11.14</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1495</th>
      <td>62</td>
      <td>5.13</td>
      <td>-5.32</td>
      <td>32.46</td>
      <td>72.04</td>
      <td>1.17</td>
      <td>-1.62</td>
      <td>0.41</td>
      <td>0.33</td>
      <td>1</td>
      <td>-306.75</td>
      <td>1.20</td>
      <td>9.24</td>
      <td>0.96</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1496</th>
      <td>30</td>
      <td>0.69</td>
      <td>-3.96</td>
      <td>32.70</td>
      <td>78.04</td>
      <td>-0.16</td>
      <td>-0.57</td>
      <td>0.01</td>
      <td>-1.67</td>
      <td>1</td>
      <td>-288.75</td>
      <td>1.48</td>
      <td>9.68</td>
      <td>-0.08</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>94</td>
      <td>4.95</td>
      <td>-5.38</td>
      <td>32.19</td>
      <td>91.04</td>
      <td>1.99</td>
      <td>1.47</td>
      <td>0.56</td>
      <td>-1.67</td>
      <td>1</td>
      <td>-252.75</td>
      <td>-0.86</td>
      <td>12.04</td>
      <td>-0.28</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1498</th>
      <td>70</td>
      <td>3.72</td>
      <td>-6.82</td>
      <td>33.06</td>
      <td>74.04</td>
      <td>0.50</td>
      <td>2.52</td>
      <td>0.24</td>
      <td>0.33</td>
      <td>1</td>
      <td>-351.75</td>
      <td>-0.08</td>
      <td>9.48</td>
      <td>0.34</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1499</th>
      <td>52</td>
      <td>3.09</td>
      <td>-10.72</td>
      <td>30.69</td>
      <td>74.04</td>
      <td>0.82</td>
      <td>0.33</td>
      <td>0.40</td>
      <td>-1.67</td>
      <td>0</td>
      <td>-252.75</td>
      <td>6.34</td>
      <td>10.02</td>
      <td>0.32</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>1500 rows × 16 columns</p>
</div>


```python
# export to csv
test_df.to_csv('CE802_P2_Results.csv', index=False)
```
</font> 
