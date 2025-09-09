# %%
#Bibliotecas necessárias
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# %%
#Carregando os datasets de treino e teste usando o pandas
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

# %%
#Apresenta dados dos dataset de treino e teste
print(trainData.info())

# %%
#Verifica valores nulos nos datasets de treino e teste
print(trainData.isnull().sum())
print("\n")
print(testData.isnull().sum())

# %%
#Mapear colunas
col = pd.Series(list(trainData.columns))
print(col)

# %%
#Funcao utilizada para preparar os datasets de treino e teste adicionando novas colunas
def build_features(X: pd.DataFrame) -> pd.DataFrame:
    sex = {'female':1, 'male':0}
    X['woman'] = X['Sex'].map(sex)
    X['Fare'] = X['Fare'].fillna(X['Fare'].mean())
    X['Age'] = X['Age'].fillna(X['Age'].mean())
    X['Embarked'] = X['Embarked'].fillna('S')
    port = {'S': 1, 'C': 2, 'Q': 3}
    X['port'] = X['Embarked'].map(port)
    X['child'] = np.where(X['Age'] < 12, 1, 0)
    return X

# %%
cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

# %%
#Ajustes dos datasets de treino e teste
y_train = trainData['Survived']

# Remover colunas que não são features brutas
X_train_raw = trainData.drop(['PassengerId','Survived'], axis=1)
X_test_raw  = testData.drop(['PassengerId'], axis=1)

# Adicionar features SEM a coluna alvo
X_train = build_features(X_train_raw.copy())
X_test  = build_features(X_test_raw.copy())

features = ['Pclass', 
            'Age', 
            'SibSp',
            'Parch', 
            'Fare', 
            'woman', 
            'port', 
            'child',
           ]
X_train = X_train[features]
X_test = X_test[features]

# %%
#Função teste para buscar os melhores hiperparametros para o modelo de Logistic Regression
def lr_train_model(parameters):
    model = LogisticRegression(
        penalty = parameters[0],        
        C = parameters[1],              
        solver = parameters[2],         
        max_iter = 1000,
        random_state = 0
    )
    score = cross_val_score(model, X_train, y_train, cv = cv)
    mean = np.mean(score)
    return -mean 

#Parametros testados
parameters = [
    ('l1', 'l2'),                     
    (0.0001, 10.0),
    ('liblinear', 'saga')
]

#chamada da funcão e resultados
otimos = gp_minimize(lr_train_model, parameters, random_state = 0, n_calls = 50, n_random_starts = 10)
print(otimos.fun, otimos.x)


# %%
#Função teste para buscar os melhores hiperparametros para o modelo de Naive Bayes Gaussiano
def nb_train_model(parameters):
    model = GaussianNB(var_smoothing = parameters[0])
    score = cross_val_score(model, X_train, y_train, cv = cv)
    mean = np.mean(score)
    return -mean 

#Parametros testados
parameters = [
    (1e-12, 1e-6)
]

#chamada da funcão e resultados
otimos = gp_minimize(nb_train_model, parameters, random_state = 0, n_calls = 50, n_random_starts = 10)
print(otimos.fun, otimos.x)

# %%
# Função teste para buscar os melhores hiperparametros para o KNN
def knn_train_model(parameters):
    model = KNeighborsClassifier(
        n_neighbors = parameters[0],
        weights = parameters[1],
        p = parameters[2]
    )
    score = cross_val_score(model, X_train, y_train, cv = cv) 
    mean = np.mean(score)
    return -mean   # gp_minimize sempre minimiza → usar negativo da acurácia média

#Parametros testados
parameters = [
    (1, 30),                         
    ('uniform', 'distance'),         
    (1, 2)
]

#chamada da funcão e resultados
otimos = gp_minimize(knn_train_model, parameters, random_state = 0, n_calls = 50, n_random_starts = 10)
print(otimos.fun, otimos.x)

# %%
# Função teste para buscar os melhores hiperparametros para o Decision Tree
def dt_train_model(parameters):
    model = DecisionTreeClassifier(
        criterion = parameters[0],
        max_depth = parameters[1],
        min_samples_split = parameters[2],
        min_samples_leaf = parameters[3],
        random_state = 0
    )
    score = cross_val_score(model, X_train, y_train, cv = cv)
    mean = np.mean(score)
    return -mean 

#Parametros testados
parameters = [
    ('gini', 'entropy'),   
    (1, 20),              
    (2, 10),               
    (1, 10)               
]

#chamada da funcão e resultados
otimos = gp_minimize(dt_train_model, parameters, random_state = 0, n_calls = 50, n_random_starts = 10)
print(otimos.fun, otimos.x)

# %%
# Função teste para buscar os melhores hiperparametros para o Random Forest
def rf_train_model(parameters):
    model = RandomForestClassifier(criterion = parameters[0], n_estimators = parameters[1], max_depth = parameters[2], min_samples_split = parameters[3], min_samples_leaf = parameters[4], random_state = 0, n_jobs = -1)
    score = cross_val_score(model, X_train, y_train, cv = cv)
    mean = np.mean(score)
    return -mean

#Parametros testados
parameters = [('entropy', 'gini'), (500, 1200), (3,20), (5,15), (1,10)]

#chamada da funcão e resultados
otimos = gp_minimize(rf_train_model, parameters, random_state = 0, n_calls = 50, n_random_starts = 10)
print(otimos.fun, otimos.x)


# %%
#Modelo de Logistic Regression para classificacao com os melhores parametros obtidos pela funcao de teste
model_lr = LogisticRegression(penalty = 'l2', C = 10, solver = 'liblinear', max_iter = 1000, random_state = 0)
score = cross_val_score(model_lr, X_train, y_train, cv = cv)

# %%
#Naive Bayes para classificacao com os melhores parametros obtidos pela funcao de teste
model_nb = GaussianNB(var_smoothing = 9.999)
score = cross_val_score(model_nb, X_train, y_train, cv = cv)

# %%
#KNN para classificacao com os melhores parametros obtidos pela funcao de teste
model_nc = KNeighborsClassifier(n_neighbors = 17, weights = 'distance', p = 1)
score = cross_val_score(model_nc, X_train, y_train, cv = cv)

# %%
#Decision tree para classificacao com os melhores parametros obtidos pela funcao de teste
model_dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 9, min_samples_split = 2, min_samples_leaf = 8, random_state = 0)
score = cross_val_score(model_dt, X_train, y_train, cv = cv)

# %%
#Random Forest
model_rf = RandomForestClassifier(criterion= "gini", n_estimators = 1137, max_depth = 17, min_samples_split = 15, min_samples_leaf = 1, random_state=0)
score = cross_val_score(model_rf, X_train, y_train, cv = cv)

# %%
#Escolhendo o melhor modelo
model_voting = VotingClassifier(estimators = [('LR', model_lr), ('NB', model_nb), ('KNN', model_nc), ('DT', model_dt), ('RF', model_rf)], voting = 'hard')
model_voting.fit(X_train, y_train)
score = cross_val_score(model_voting, X_train, y_train, cv = cv)
print(np.mean(score))

# %%
#Salvano o modelo para submeter ao kaggle
y_pred = model_voting.predict(X_test)
submission = pd.DataFrame(testData['PassengerId'])
submission['Survived'] = y_pred
submission.to_csv('submission15.csv', index = False)


