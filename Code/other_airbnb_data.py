# %%
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.linear_model import ElasticNet
# from sklearn.svm import SVR
from xgboost import XGBRegressor
# from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import ttest_ind, f_oneway 
from feature_engine.encoding import RareLabelEncoder

# %%
data = pd.read_csv("train.csv")
# %%
data
# %%
data.columns
# %%
data.isna().sum()
# %%
main_label = 'price'
# Exclude 1% of smallest and 1% of highest prices
P = np.percentile(data[main_label], [1, 99])
df = data[(data[main_label] > P[0]) & (data[main_label] < P[1])]
# combine neighbourbood and neighbourhood_group
df['neighbourhood'] = df['neighbourhood'] + ', ' + df['neighbourhood_group']
# log10-transform columns and group for larger bins
for col in ['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']:
    df[f'log10_{col}'] = df[col].apply(lambda x: 1/5*round(5*np.log10(1+x)))
    df = df.drop([col], axis=1)
# set up the rare label encoder limiting number of categories to max_n_categories
for col in ['neighbourhood', 'room_type']:
    encoder = RareLabelEncoder(n_categories=1, max_n_categories=70, replace_with='Other', tol=20/df.shape[0])
    df[col] = encoder.fit_transform(df[[col]])
# drop unused columns
cols2drop = ['name', 'host_id', 'host_name', 'latitude', 'longitude', 'neighbourhood_group', 
             'last_review', 'reviews_per_month']
df = df.drop(cols2drop, axis=1)
print(df.shape)
df.sample(5).T

# %%'amenities',
X = df[['neighbourhood','room_type', 'log10_minimum_nights','log10_number_of_reviews','log10_calculated_host_listings_count','log10_availability_365']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OneHotEncoder(), [ 'neighbourhood', 'room_type'])
    ])

# Pipeline
baseline_model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

baseline_model.fit(X_train, y_train)
y_pred = baseline_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print('baseline_model RMSE:', rmse)
r2 = r2_score(y_test,y_pred)
print("R-squared:", r2)

# %%
# decision_tree baseline model

# Define the preprocessor as part of the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OneHotEncoder(), [ 'neighbourhood', 'room_type'])
    ])

# Pipeline with decision tree regressor
tree_model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', DecisionTreeRegressor())])

# Parameter grid for GridSearchCV
param_grid = {
    'regressor__max_depth': [2, 5, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(tree_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best estimator found by grid search
print('Best parameters found:', grid_search.best_params_)

# Predict using the best model
y_pred = grid_search.predict(X_test)

# Calculate RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print('Decision Tree RMSE:', rmse)
print('R-squared:', r2)
# %%
# %%

# Define the preprocessor as before
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OneHotEncoder(), [ 'neighbourhood', 'room_type'])
    ])

# Define the pipeline with a random forest regressor
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor())])

# Define the parameter grid for the RandomForestRegressor
param_grid = {
    'regressor__n_estimators': [100,200,300],  # The number of trees in the forest
    'regressor__max_depth': [3,4,5,6,7,8,10,20],  # The maximum depth of the trees
}

# Create the GridSearchCV object
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best estimator
print('Best parameters found:', grid_search.best_params_)

# Predict using the best model
y_pred = grid_search.predict(X_test)

# Calculate RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print('Random Forest RMSE:', rmse)
print('R-squared:', r2)

# %%

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OneHotEncoder(), [ 'neighbourhood', 'room_type'])
    ])

# Pipeline with XGBoost regressor
xgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', XGBRegressor())])

# Parameter grid for XGBoost
param_grid = {
    'regressor__n_estimators': [100, 200],  # Number of gradient boosted trees. Equivalent to number of boosting rounds
    'regressor__learning_rate': [0.01, 0.1],  # Boosting learning rate (xgb's "eta")
    'regressor__max_depth': [3, 6],  # Maximum depth of a tree
    'regressor__subsample': [0.7, 1],  # Subsample ratio of the training instances
    'regressor__colsample_bytree': [0.7, 1]  # Subsample ratio of columns when constructing each tree
}

# Create the GridSearchCV object
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best estimator found by grid search
print('Best parameters found:', grid_search.best_params_)

# Predict using the best model
y_pred = grid_search.predict(X_test)

# Calculate RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print('XGBoost Regressor RMSE:', rmse)
print('R-squared:', r2)

# %%
from sklearn.linear_model import Lasso

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OneHotEncoder(), [ 'neighbourhood', 'room_type'])])

# Define the pipeline with Lasso regression
lasso_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', Lasso())])

# Define the parameter grid to search
param_grid = {
    'regressor__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(lasso_pipeline, param_grid, cv=5)

# Fit the model
grid_search.fit(X_train, y_train)

# Predict and evaluate
y_pred_lasso = grid_search.predict(X_test)

rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

print('Lasso Model RMSE:', rmse_lasso)
print("Lasso Model R-squared:", r2_lasso)
# %%
from sklearn.svm import SVR

# Define the preprocessor as before
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OneHotEncoder(), [ 'neighbourhood', 'room_type'])
    ])

# Pipeline with SVM regressor
svm_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', SVR())])

# Parameter grid for SVR
param_grid = {
    'regressor__C': [0.1, 1, 10],  # Regularization parameter
    'regressor__epsilon': [0.01, 0.1, 0.5],  # Epsilon in the epsilon-SVR model
    'regressor__kernel': ['linear', 'poly', 'rbf', 'sigmoid']  # Type of kernel function
}

# Create the GridSearchCV object
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best estimator found by grid search
print('Best parameters found:', grid_search.best_params_)

# Predict using the best model
y_pred = grid_search.predict(X_test)

# Calculate RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print('SVM Regressor RMSE:', rmse)
print('R-squared:', r2)

# %%
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

# initialize data
y = df[main_label].values.reshape(-1,)
X = df.drop([main_label], axis=1)
cat_cols = df.select_dtypes(include=['object']).columns
cat_cols_idx = [list(X.columns).index(c) for c in cat_cols]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.5, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# initialize Pool
train_pool = Pool(X_train, 
                  y_train, 
                  cat_features=cat_cols_idx)
test_pool = Pool(X_test,
                 y_test,
                 cat_features=cat_cols_idx)
# specify the training parameters 
model = CatBoostRegressor(iterations=1000, 
                          depth=5,
                          verbose=0,
                          learning_rate=0.01, 
                          loss_function='RMSE')
# train the model
model.fit(train_pool)

# Predict using the best model
y_pred = model.predict(X_test)

# Calculate RMSE and R-squared
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print('catboost Regressor RMSE:', rmse)
print('R-squared:', r2)

# %%
