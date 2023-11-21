#%% [markdown]
# This is the project code for Team 3

#%%Initalising and loading the data set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns # if you (optionally) want to show fancy plots
import rfit 
import geopandas as gpd

#loading dataset
data = pd.read_csv("Listings.csv", encoding='unicode_escape')

#Basic information about the dataset
print(data.info())

#first few rows of the dataset
print("\nFirst Few Rows of the Dataset:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop rows with missing values
data_cleaned = data.dropna()



#%%
# summary after cleaning
print(data_cleaned.info())





# %% EDA on the cleaned data

# Display basic statistics
print("Basic Statistics:")
print(data_cleaned.describe())

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
    plt.title(f'Boxplot of Price by {var}')
    plt.show()
    
    
    
    
    
# %%
from shapely.geometry import Point
geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
gdf = gpd.GeoDataFrame(data, geometry=geometry)


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

fig, ax = plt.subplots(figsize=(12, 8))
world.plot(ax=ax, color='lightgrey')


gdf.plot(ax=ax, markersize=5, color='red', alpha=0.5)

plt.title('Geographical Distribution of Property Prices')
plt.show()




# %%
import folium
mean_lat = data_cleaned['latitude'].mean()
mean_lon = data_cleaned['longitude'].mean()
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, control_scale=True)
for index, row in data_cleaned.iterrows():
    folium.Marker([row['latitude'], row['longitude']], popup=f"Price: ${row['price']}").add_to(m)

m.save('property_map.html')
# %%
plt.figure(figsize=(12, 8))
sns.scatterplot(x='neighbourhood', y='price', data=data_cleaned)
plt.title('Price Variation Across Neighborhoods')
plt.show()





# %%
# Explore the distribution of superhosts in different districts
plt.figure(figsize=(12, 8))
sns.countplot(x='neighbourhood', hue='host_is_superhost', data=data_cleaned)
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Superhosts Across Neighborhoods')
plt.show()



# %%
# Filter data for districts with more superhosts
superhost_districts = data_cleaned[data_cleaned['host_is_superhost'] == 't']

# Visualize pricing strategy and customer satisfaction
plt.figure(figsize=(12, 8))
sns.boxplot(x='neighbourhood', y='price', hue='review_scores_value', data=superhost_districts)
plt.xticks(rotation=45, ha='right')
plt.title('Pricing and Satisfaction in Districts with More Superhosts')
plt.show()





# %% To try answering smart questions
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


price_neigh_cols = ['price', 'neighbourhood', 'district', 'accommodates', 'bedrooms', 'amenities', 'host_is_superhost', 'review_scores_rating']
price_neigh_df = data_cleaned[price_neigh_cols]

price_neigh_df = pd.get_dummies(price_neigh_df, columns=['neighbourhood', 'district', 'amenities', 'host_is_superhost'])

X = price_neigh_df.drop('price', axis=1)
y = price_neigh_df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# %% result from lenier model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')



plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices')
plt.show()

#not a good fit
# %%Random forest

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
# %%

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# %%
feature_importances = model.feature_importances_
feature_names = X_train.columns

indices = np.argsort(feature_importances)[::-1]

for f in range(len(feature_names)):
    print(f"{f + 1}. {feature_names[indices[f]]}: {feature_importances[indices[f]]}")

plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_names)), feature_importances[indices], align="center")
plt.xticks(range(len(feature_names)), feature_names[indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.show()
# %%2. lm 

amenities_price_cols = ['price', 'amenities', 'neighbourhood', 'accommodates', 'bedrooms', 'review_scores_rating']

amenities_price_df = data_cleaned[amenities_price_cols]

amenities_price_df = pd.get_dummies(amenities_price_df, columns=['amenities', 'neighbourhood'])

X_amenities = amenities_price_df.drop('price', axis=1)
y_amenities = amenities_price_df['price']
X_train_amenities, X_test_amenities, y_train_amenities, y_test_amenities = train_test_split(X_amenities, y_amenities, test_size=0.2, random_state=42)

model_amenities = LinearRegression()
model_amenities.fit(X_train_amenities, y_train_amenities)


y_pred_amenities = model_amenities.predict(X_test_amenities)

mse_amenities = mean_squared_error(y_test_amenities, y_pred_amenities)
print(f'Mean Squared Error for Price Prediction: {mse_amenities}')

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_amenities, y=y_pred_amenities)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices (Amenities)')
plt.show()

# %% 3.

from scipy.stats import ttest_ind, f_oneway 

plt.figure(figsize=(12, 8))
sns.countplot(x='district', hue='host_is_superhost', data=data_cleaned)
plt.title('Distribution of Superhosts Across Districts')
plt.show()

superhost_prices = data_cleaned[data_cleaned['host_is_superhost'] == 't']['price']
non_superhost_prices = data_cleaned[data_cleaned['host_is_superhost'] == 'f']['price']
 
 
t_stat, p_value = ttest_ind(superhost_prices, non_superhost_prices)
print("T-test Results:")
print("T-statistic:", t_stat)
print("P-value:", p_value)

anova_results = f_oneway(data_cleaned[data_cleaned['host_is_superhost'] == 't']['review_scores_rating'],
                         data_cleaned[data_cleaned['host_is_superhost'] == 'f']['review_scores_rating'])
print("\nANOVA Results:")
print("F-statistic:", anova_results.statistic)
print("P-value:", anova_results.pvalue)

#The T-test results indicate a statistically significant difference in pricing between districts with and without superhosts
#The ANOVA results indicate a statistically significant difference in customer satisfaction between districts with and without superhosts
#Districts with superhosts tend to have both higher prices and higher customer satisfaction
# %%
