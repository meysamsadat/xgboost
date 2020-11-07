import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
df = pd.read_csv(r'/home/meysam/Desktop/data projects/final_hotel.csv')

df.drop(df.columns[0],axis=1,inplace=True)
x = df.iloc[:, df.columns != 'is_canceled']
y = df.is_canceled

model = XGBClassifier()
parameters ={
'n_estimators' : [100,250,500],
'learning_rate' : [0.01, 0.1],
'subsample' :[0.5, 1.0],
'max_depth' : [3,5,7],
'objective':['binary:logistic']
}

grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring='f1', verbose=True, n_jobs=-1)
grid_search.fit(x, y)
print(grid_search.best_score_)
print(grid_search.best_params_)
