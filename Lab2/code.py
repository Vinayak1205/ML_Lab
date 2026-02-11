import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the dataset
url = "https://media.geeksforgeeks.org/wp-content/uploads/20240319120216/housing.csv"
housing = pd.read_csv(url)

# --- 1. Describe and Info ---
print(housing.info())
print(housing.describe())

# --- 2. Histograms ---
housing.hist(bins=50, figsize=(20,15))
plt.show()

# --- 3. Stratified Split ---
# Create an income category to sample representatively
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Remove income_cat so the data is back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# --- 4. Geographical Visualization ---
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()

# --- 5. Correlation ---
# Calculate correlation with house value
corr_matrix = housing.drop("ocean_proximity", axis=1).corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Plotting the strongest correlate: Median Income
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.show()

# --- 6, 10. Custom Transformer & Pipeline ---
# Index of columns for the transformer
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self 
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# --- 7, 8, 10. Building the Full Pipeline ---
housing_num = strat_train_set.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")), # Step 7: Cleaning
    ('attribs_adder', CombinedAttributesAdder()),   # Step 6: Combinations
    ('std_scaler', StandardScaler()),              # Step 9: Scaling
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),         # Step 8: Encoding
])

# Execute Pipeline
housing_prepared = full_pipeline.fit_transform(strat_train_set)

# --- Output Results ---
print("\nShape of prepared data:", housing_prepared.shape)
print("First row of prepared data:\n", housing_prepared[0])
