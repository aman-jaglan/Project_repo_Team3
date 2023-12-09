# code for modeling
# %% [markdown]
# This is the project code for Team 3

#%%Initalising and loading the data set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
from plotly.subplots import make_subplots
from shapely.geometry import Point
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import ttest_ind, f_oneway 

# %%
data = pd.read_csv("Listings.csv", encoding='unicode_escape', low_memory=False)

data.head(5)
# %%
data.columns
# %%
data.isna().sum()
# %%
data.shape
# %%
columns_to_drop = ['host_response_time','district','host_response_rate','host_acceptance_rate']
data.drop(columns = columns_to_drop, inplace = True)
# %%
def fill_null_with_rounded_average(data, column_name):
    """
    Fill null values in the specified column with the rounded average.

    Parameters:
    - data: DataFrame
    - column_name: str, the name of the column with null values to be filled

    Returns:
    - None (modifies the DataFrame in place)
    """
    average_value = data[column_name].mean()
    data[column_name].fillna(average_value, inplace=True)
    data[column_name] = data[column_name].round()

fill_null_with_rounded_average(data, 'review_scores_rating')
fill_null_with_rounded_average(data, 'review_scores_accuracy')
fill_null_with_rounded_average(data, 'review_scores_cleanliness')
fill_null_with_rounded_average(data, 'review_scores_checkin')
fill_null_with_rounded_average(data, 'review_scores_communication')
fill_null_with_rounded_average(data, 'review_scores_location')
fill_null_with_rounded_average(data, 'review_scores_value')
# %%
data_cleaned = data.dropna()
# %%
data_cleaned.head(5)
# %%
data_cleaned.info()

# %%
# Display basic statistics
print("Basic Statistics:")
print(data_cleaned.describe())

# %%
data_cleaned.to_csv('air_bnb.csv')

# %%
# the corr heatmap
corr_all = data_cleaned.corr(numeric_only=True)
corr_all['price']

# %%
# EDA
sns.set(font_scale=1.5)

fig, ax = plt.subplots(figsize=(30, 30))
sns.heatmap(corr_all, cmap="PuBuGn", annot=True, linewidths=.5, ax=ax)

ax.set_title("Correlation Heatmap")

plt.show()

# %%
city_hotel_counts = data_cleaned['city'].value_counts().reset_index()
city_hotel_counts.columns = ['city', 'number_of_hotels']

fig = px.bar(city_hotel_counts, x='number_of_hotels', y='city', orientation='h',
             title='Number of Hotels in Each City', text='number_of_hotels',
            color_discrete_sequence=px.colors.qualitative.Pastel)


fig.update_layout(yaxis=dict(categoryorder='total ascending'), xaxis_title='Number of Hotels')

fig.show()

# %%
fig = px.scatter_mapbox(data_cleaned, lat='latitude', lon='longitude', text='city', hover_name='city', size='price',
                        size_max=40, zoom=1, title='Price Comparison and distribution of hotels across cities')

fig.update_layout(mapbox_style="carto-positron", showlegend=False, height=600)

fig.show()
# %%
filtered_df = data_cleaned[data_cleaned['price'] > 100000]
fig = px.treemap(
    filtered_df,
    path=['city', 'name', 'amenities'],
    values='price',
    hover_data=['neighbourhood'],
    title='Hotels which have Price more than > 100000',
    color_discrete_sequence=px.colors.diverging.RdBu,
)


fig.update_traces(
    hovertemplate='City: %{id}<br>Hotel: %{label}<br>Price: $%{value:.2f}',
)
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))

fig.show()
# %%
highPrice = data_cleaned[data_cleaned['price'] > 100000]
lowPrice = data_cleaned[data_cleaned['price'] < 1000]


# Assuming df_high_price and df_low_price are your DataFrames with prices > 100k and < 1k respectively
common_neighborhoods = set(highPrice['neighbourhood']).intersection(set(lowPrice['neighbourhood']))
common_neighborhood_df = highPrice[highPrice['neighbourhood'].isin(common_neighborhoods)]
neighborhood_counts = common_neighborhood_df['neighbourhood'].value_counts().reset_index().rename(columns={'index': 'neighbourhood', 'neighbourhood': 'count'})
neighborhood_counts.columns = ['neighbourhood', 'count']

styled_neighborhood_counts = neighborhood_counts.style.background_gradient(axis=0, cmap='YlOrBr', subset=["count"])

styled_neighborhood_counts = neighborhood_counts.style.bar(
    subset=["count"],
    cmap='RdBu',  
).set_table_styles([
    {
        'selector': 'table',
        'props': [
            ('border-collapse', 'collapse'),
            ('border', '2px solid black'),
            ('background-color', '#f2f2f2'),
        ],
    },
    {
        'selector': 'th, td',
        'props': [('border', '1px solid black')],
    },
]).set_properties(**{'font-weight': 'bold'})

styled_neighborhood_counts

# %%
# Pairplot for selected numerical variables
numerical_vars = ['accommodates', 'bedrooms', 'minimum_nights', 'review_scores_rating']
sns.pairplot(data_cleaned[numerical_vars])
plt.suptitle('Pairplot of Selected Numerical Variables', y=1.02)
plt.show()

# Boxplot for categorical variables
categorical_vars = ['room_type', 'host_is_superhost', 'instant_bookable']
for var in categorical_vars:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=var, y='price', data=data_cleaned)
    plt.yscale('log') # using log scaled y axis to have a good look on boxplot
    plt.title(f'Boxplot of Price by {var}')
    plt.show()

# %%

geometry = [Point(xy) for xy in zip(data_cleaned['longitude'], data_cleaned['latitude'])]
gdf = gpd.GeoDataFrame(data_cleaned, geometry=geometry)


world = gpd.read_file('ne_110m_admin_0_countries.shp')

fig, ax = plt.subplots(figsize=(12, 8))
world.plot(ax=ax, color='lightgrey')


gdf.plot(ax=ax, markersize=5, color='red', alpha=0.5)

plt.title('Geographical Distribution of Property Prices')
plt.show()

# %%
### Too big to open the html file
# mean_lat = data_cleaned['latitude'].mean()
# mean_lon = data_cleaned['longitude'].mean()
# m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, control_scale=True)
# for index, row in data_cleaned.iterrows():
#     folium.Marker([row['latitude'], row['longitude']], popup=f"Price: ${row['price']}").add_to(m)

# m.save('property_map.html')

# %%
### The x labels are messed up
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.figure(figsize=(12, 8))
# sns.scatterplot(x='neighbourhood', y='price', data=data_cleaned)
# plt.title('Price Variation Across Neighborhoods')
# plt.show()

# # %%
# # Explore the distribution of superhosts in different districts
# plt.figure(figsize=(12, 8))
# sns.countplot(x='neighbourhood', hue='host_is_superhost', data=data_cleaned)
# plt.xticks(rotation=45, ha='right')
# plt.title('Distribution of Superhosts Across Neighborhoods')
# plt.show()

# # %%
# # Filter data for districts with more superhosts
# superhost_districts = data_cleaned[data_cleaned['host_is_superhost'] == 't']

# # Visualize pricing strategy and customer satisfaction
# plt.figure(figsize=(12, 8))
# sns.boxplot(x='neighbourhood', y='price', hue='review_scores_value', data=superhost_districts)
# plt.xticks(rotation=45, ha='right')
# plt.title('Pricing and Satisfaction in Districts with More Superhosts')
# plt.show()

# %%
# New York geopandas 
# Can only do New York because it is hard to find other city shapefiles

# Load New York geographic data
ny_zone = gpd.read_file('new_york_boundries.shp')
ny_zone.crs = "EPSG:4326"

# Add a unique identifier to each zone
ny_zone['POLY_ID'] = range(1, len(ny_zone) + 1)

# Prepare the Airbnb data
geometry = [Point(xy) for xy in zip(data_cleaned['longitude'], data_cleaned['latitude'])]
airbnb_gdf = gpd.GeoDataFrame(data_cleaned, geometry=geometry)
airbnb_gdf.crs = "EPSG:4326"

# Perform Spatial Join using 'predicate' instead of 'op'
airbnb_zone = gpd.sjoin(airbnb_gdf, ny_zone, how="inner", predicate="within")

# Calculate Average Prices by the 'POLY_ID', explicitly setting numeric_only
meanprice = airbnb_zone.groupby('POLY_ID').mean(numeric_only=True)['price']
meanprice = meanprice.to_frame()

# Merge the average prices back into the ny_zone GeoDataFrame
final_zone = ny_zone.merge(meanprice, how='left', on='POLY_ID')
final_zone['price'].fillna(final_zone['price'].mean(), inplace=True)

# Plot the distribution of Average Price in New York
def pricemap_ny():
    f, ax = plt.subplots(figsize=(10, 7))
    final_zone.plot(ax=ax, column='price', cmap='OrRd', scheme='quantiles', legend=True)
    ax.axis('off')
    ax.set_title('New York Airbnb Average Price Distribution', fontdict={'fontsize': '20', 'fontweight': '3'})
    # Adjust the legend size and location
    leg = ax.get_legend()
    leg.set_bbox_to_anchor((0, 1))  # Moves the legend to the top left
    leg.set_frame_on(True)  # Optionally add a frame around the legend
    leg.set_title("Average Price ($)")
    plt.show()

pricemap_ny()

# %%
#plot the loction of airbnb new york
def location():
  f,ax = plt.subplots(figsize=(10,7))

  ny_zone.plot(ax=ax, facecolor='gray')

  airbnb_zone.plot(ax=ax, color='yellow', markersize=0.04)

  plt.legend(['Airbnb Hotel location'])

location()

# %%
# Now it is time for modeling!
# We need to conduct a linear regression as a baseline

# %%
data_cleaned
# %%
data_cleaned.columns
# %%
# data_cleaned['instant_bookable'].value_counts()
# paris_data = data_cleaned[data_cleaned['city'] == 'Paris']
# paris_data['neighbourhood'].value_counts()
# %%'amenities',
X = data_cleaned[['host_is_superhost','room_type', 'accommodates','bedrooms','amenities','review_scores_accuracy','review_scores_location',
                'minimum_nights','host_identity_verified', 'review_scores_value','instant_bookable','city',
                'host_total_listings_count']]
y = data_cleaned['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train

# %%

text_features = ['amenities']
# Create a pipeline for text processing
text_ff = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svd', TruncatedSVD(n_components=10))
])

# Process the text feature on training data
text_train = text_ff.fit_transform(X_train[text_features[0]].astype(str))
text_train = pd.DataFrame(text_train, columns=[f'text{i}' for i in range(10)])

# Process the text feature on test data using the fitted pipeline
text_test = text_ff.transform(X_test[text_features[0]].astype(str))
text_test = pd.DataFrame(text_test, columns=[f'text{i}' for i in range(10)])

# Reset index before merging
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Merge the new text features
X_train = X_train.merge(text_train, left_index=True, right_index=True)
X_test = X_test.merge(text_test, left_index=True, right_index=True)

# Drop the original 'amenities' column
X_train.drop('amenities', axis=1, inplace=True)
X_test.drop('amenities', axis=1, inplace=True)

# %%
# linear_regression baseline model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['accommodates', 'bedrooms','minimum_nights','review_scores_value','host_total_listings_count','review_scores_accuracy','review_scores_location']),
        ('ord', OneHotEncoder(), [ 'host_is_superhost', 'room_type','instant_bookable','host_identity_verified','city'])
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
        ('num', StandardScaler(), ['accommodates', 'bedrooms','minimum_nights','review_scores_value','host_total_listings_count','review_scores_accuracy','review_scores_location']),
        ('ord', OneHotEncoder(), [ 'host_is_superhost', 'room_type','instant_bookable','host_identity_verified','city'])
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

# Define the preprocessor as before
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['accommodates', 'bedrooms','minimum_nights','review_scores_value','host_total_listings_count','review_scores_accuracy','review_scores_location']),
        ('ord', OneHotEncoder(), [ 'host_is_superhost','room_type','instant_bookable','host_identity_verified','city'])
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
# from sklearn.svm import SVR

# # Define the preprocessor as before
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), ['accommodates', 'bedrooms', 'minimum_nights', 'review_scores_accuracy', 'review_scores_value', 'review_scores_location']),
#         ('ord', OneHotEncoder(), ['host_is_superhost', 'city', 'room_type'])
#     ])

# # Pipeline with SVM regressor
# svm_model = Pipeline(steps=[('preprocessor', preprocessor),
#                             ('regressor', SVR())])

# # Parameter grid for SVR
# param_grid = {
#     'regressor__C': [0.1, 1, 10],  # Regularization parameter
#     'regressor__epsilon': [0.01, 0.1, 0.5],  # Epsilon in the epsilon-SVR model
#     'regressor__kernel': ['linear', 'poly', 'rbf', 'sigmoid']  # Type of kernel function
# }

# # Create the GridSearchCV object
# grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Best estimator found by grid search
# print('Best parameters found:', grid_search.best_params_)

# # Predict using the best model
# y_pred = grid_search.predict(X_test)

# # Calculate RMSE and R-squared
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# print('SVM Regressor RMSE:', rmse)
# print('R-squared:', r2)

# %%

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['accommodates', 'bedrooms','minimum_nights','review_scores_value','host_total_listings_count','review_scores_accuracy','review_scores_location']),
        ('ord', OneHotEncoder(), [ 'host_is_superhost', 'room_type','instant_bookable','host_identity_verified','city'])
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
data = pd.read_csv("Listings.csv", encoding='unicode_escape', low_memory=False)
plt.figure(figsize=(12, 8))
sns.countplot(x='district', hue='host_is_superhost', data=data)
plt.title('Distribution of Superhosts Across Districts')
plt.show()

superhost_prices = data[data['host_is_superhost'] == 't']['price']
non_superhost_prices = data[data['host_is_superhost'] == 'f']['price']
 
 
t_stat, p_value = ttest_ind(superhost_prices, non_superhost_prices)
print("T-test Results:")
print("T-statistic:", t_stat)
print("P-value:", p_value)

anova_results = f_oneway(data[data['host_is_superhost'] == 't']['review_scores_rating'],
                         data[data['host_is_superhost'] == 'f']['review_scores_rating'])
print("\nANOVA Results:")
print("F-statistic:", anova_results.statistic)
print("P-value:", anova_results.pvalue)

#The T-test results indicate a statistically significant difference in pricing between districts with and without superhosts
#The ANOVA results indicate a statistically significant difference in customer satisfaction between districts with and without superhosts
#Districts with superhosts tend to have both higher prices and higher customer satisfaction
# %%
