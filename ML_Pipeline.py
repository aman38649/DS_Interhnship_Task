import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Step 1: Read the JSON file (Not shown in this code as it's assumed you already have the JSON data)
with open("algoparams_from_ui.json") as json_file:
    algoparams_from_ui = json.load(json_file)

# Step 2: Read the Iris dataset
data = pd.read_csv("iris_dataset.csv")

# Extract target column and prediction type
target_column = "species"
prediction_type = "classification"  # Assuming it's classification since the target is species

# Step 3: Perform feature handling
# For this example, we'll only apply imputation as no reduction is selected in the JSON

for feature_details in algoparams_from_ui["feature_handling"]:
    if feature_details["is_selected"]:
        feature_name = feature_details["feature_name"]
        if feature_details["feature_variable_type"] == "numerical":
            # Apply numerical feature handling (only imputation in this case)
            if feature_details["numerical_handling"] == "Impute":
                impute_value = feature_details["impute_value"]
                data[feature_name] = data[feature_name].fillna(impute_value)

# Step 4: Feature reduction

num_of_features_to_keep = algoparams_from_ui["feature reduction"]["Principal Component Analysis"]["num_of_features_to_keep"]

# Perform PCA
pca = PCA(n_components=num_of_features_to_keep)
data_features = data.drop(columns=[target_column])  # Remove the target column from features
data_features_reduced = pca.fit_transform(data_features)
data_features_reduced = pd.DataFrame(data_features_reduced, columns=[f"PC{i+1}" for i in range(num_of_features_to_keep)])
data = pd.concat([data[target_column], data_features_reduced], axis=1)

if algoparams_from_ui["feature reduction"]["No Reduction"]["is_selected"]:
    # No reduction selected, no changes needed to the features
    pass
elif algoparams_from_ui["feature reduction"]["Tree-based"]["is_selected"]:
    # Tree-Based feature reduction
    num_of_features_to_keep = algoparams_from_ui["feature reduction"]["Tree-based"]["num_of_features_to_keep"]
    depth_of_trees = algoparams_from_ui["feature reduction"]["Tree-based"]["depth of trees"]
    num_of_trees = algoparams_from_ui["feature reduction"]["Tree-based"]["num_of_trees"]

    tree_based_selector = SelectKBest(score_func=f_regression, k=num_of_features_to_keep)
    features = data.drop(columns=[target_column])
    target = data[target_column]
    features_reduced = tree_based_selector.fit_transform(features, target)
    selected_feature_indices = tree_based_selector.get_support(indices=True)
    selected_features = features.columns[selected_feature_indices]
    data = pd.concat([data[target_column], features[selected_features]], axis=1)

# Step 5: Create model objects and hyperparameter tuning using GridSearchCV
if prediction_type == "regression":
    models = [
        ("LinearRegression", LinearRegression()),
        ("RandomForestRegressor", RandomForestRegressor())
    ]
elif prediction_type == "classification":
    models = [
        ("LogisticRegression", LogisticRegression()),
        ("RandomForestClassifier", RandomForestClassifier()),
        ("KNeighborsClassifier", KNeighborsClassifier()),
        ("SVC", SVC())
    ]

# Step 6: Run the fit and predict on each model using GridSearchCV for hyperparameter tuning
for model_name, model in models:
    param_grid = algoparams_from_ui[model_name]  # Get the hyperparameter grid for the specific model
    grid_search = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(), scoring='accuracy', n_jobs=-1)
    features = data.drop(columns=[target_column])
    target = data[target_column]
    grid_search.fit(features, target)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(features)

    # Step 7: Log standard model metrics
    accuracy = accuracy_score(target, predictions)
    print(f"Model: {model_name}, Accuracy: {accuracy}")
