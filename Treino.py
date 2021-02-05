import pandas as pd
import calendar
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import FuncPuc1 as puc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('C:\\Users\\paulo.abreu\\Downloads\\archive\\Data_Train.xlsx')

puc.pre_p(df)
puc.feature_bin(df)

df1 = df.copy()
df1 = df1.dropna()
df1 = df1.drop(['Date_of_Journey', 'Dep_Time' , 'Arrival_Time', 'Duration', 'Total_Stops'], axis =1)

numeric_features = ['Dep_t_hour_bin', 'Date_of_J_day_bin', 'lag_bin'] #'Duration', 'Total_Stops'

categorical_features = ['Airline', 'Source' , 'Destination', 'Route', 'Additional_Info', 'Date_of_Journey_Month']

df_test = df1[0 : 5000]
df1 = df1[5000:]
df_test_y  = df_test['Price']
df_test_X = df_test.drop(['Price'], axis=1)

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestRegressor(n_estimators=200, 
                                  random_state=10, 
                                    
                                  #class_weight= 'unbalanced', 
                                  max_features=0.2, 
                                  min_samples_leaf=1, 
                                  min_samples_split=2))])

y = df1['Price']
X = df1.drop(['Price'], axis =1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

clf.fit(X_train, y_train)
print("Score do modelo (validação): %.3f" % clf.score(X_test, y_test))#X_test, y_test

print("Score do modelo (Teste): %.3f" % clf.score(df_test_X, df_test_y))#X_test, y_test

preds = clf.predict(df_test_X)

rmse = np.sqrt(mean_squared_error(df_test_y, preds))
print("RMSE: %f" % (rmse))

mae = np.sqrt(mean_absolute_error(df_test_y, preds))
print("MAE: %f" % (mae))
#clf.score(df_test_y, preds)

kfold = KFold(n_splits=10, random_state=7)

scores = cross_val_score(clf, df_test_X, df_test_y, cv=kfold)

print('Score Validação cruzada: %.3f (%.3f)' % (mean(scores), std(scores)))

filename = 'modelo_final.sav'
pickle.dump(clf, open(filename, 'wb'))


