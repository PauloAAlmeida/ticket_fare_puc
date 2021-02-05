import pickle
import pandas as pd
import FuncPuc1 as puc

df = pd.read_excel('C:\\Users\\paulo.abreu\\Downloads\\archive\\Data_Train.xlsx')
puc.pre_p(df)
puc.feature_bin(df)
df1 = df.copy()
df1 = df1.dropna()
df1 = df1.drop(['Date_of_Journey', 'Dep_Time' , 'Arrival_Time', 'Duration', 'Total_Stops'], axis =1)

df_test = df1[0 : 5000]
df1 = df1[1000:]
df_test_y  = df_test['Price']
df_test_X = df_test.drop(['Price'], axis=1)

filename = 'C:\\Users\\paulo.abreu\\Downloads\\finalized_model.sav'
#pickle.dump(clf, open(filename, 'wb'))
modelo = pickle.load(open(filename, 'rb'))
result = modelo.score(df_test_X, df_test_y)
print(result)

previsoes = modelo.predict(df_test_X)
df_previ = pd.DataFrame(previsoes)
df_final = df_test_X.merge(df_previ, left_index=True, right_index=True, how='left')

