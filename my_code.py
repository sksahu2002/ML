import pandas as pd  
from dython.nominal import associations
from dython.nominal import identify_nominal_columns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import multiprocessing

dataset = pd.read_csv(r'C:\Users\sahus\Desktop\ML_problemstatement_IDFC\case_study_devdata (1).csv')
target = dataset['target_variable']
dataset.drop('target_variable', axis=1, inplace=True)
dataset.drop('primary_key', axis=1, inplace=True)
dataset.drop('merchant_name', axis=1, inplace=True)

categorical_features=identify_nominal_columns(dataset)
print(categorical_features)
cols = len(dataset.axes[1])  

# complete_correlation= associations(dataset, filename= 'complete_correlation.png', figsize=(5,345))
# dataset_complete_corr=complete_correlation['corr']
# dataset_complete_corr.dropna(axis=1, how='all').dropna(axis=0, how='all').style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)

# # fs = SelectKBest(score_func=chi2, k='all')
# # what are scores for the features
# # for i in range(len(fs.scores_)):
# #  print('Feature %d: %f' % (i, fs.scores_[i]))
# # plot the scores
# # pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# # pyplot.show()
# target_variable = dataset['target_variable']
# input_features = dataset.drop('target_variable', axis=1, inplace=True)


# # Convert input features to a 2D array
# X = input_features.values

# # Instantiate the SelectKBest class with f_classif scoring function
# selector = SelectKBest(score_func=f_classif, k=200)  # Select top 5 features

# # Perform feature selection
# X_selected = selector.fit_transform(X, target_variable)

# # Get the support mask indicating the selected features
# selected_features = selector.get_support()

# # Print the selected feature indices
# print("Selected Feature Indices:", np.where(selected_features)[0])

# # Print the scores of the features
# print("Feature Scores:", selector.scores_)

categorical_features = dataset[['merchant_country', 'Merchant_category', 'product']]
# print(categorical_features.shape)

#performing OneHot Encoding on Catagorical_features 

# Creating an instance of the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Fitting and transforming the categorical features
encoded_categorical_features = encoder.fit_transform(categorical_features)

# Create a new df from the encoded features
encoded_categorical_df = pd.DataFrame(encoded_categorical_features, columns=encoder.get_feature_names_out(categorical_features.columns))

# Print the encoded df
# print(encoded_categorical_df)
# print(encoded_categorical_df.shape)


dataset.drop('merchant_country', axis=1, inplace=True)
dataset.drop('Merchant_category', axis=1, inplace=True)
dataset.drop('product', axis=1, inplace=True)

#Combining the encoded categorical df with the numerical df
encoded_df = dataset.join(encoded_categorical_df)

# for column in encoded_df.columns:
#     unique_values = encoded_df[column].unique()
#     print(f"Column '{column}': {unique_values}")
#     pass 
#     print(encoded_df.shape)

# Finding Correlation matrix of encoded_df
corr_matrix = encoded_df.corr()
# print(corr_matrix)

# Creating a correlation matrix plot

# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# pyplot.show()

# Forming HeatMap using Correlation matrix

# pyplot.figure(figsize=(500,442))  
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# pyplot.title('Correlation Heatmap')
# pyplot.show()

# Setting the correlation threshold
threshold = 0.80

# Finding the feature pairs with correlation above the threshold
highly_correlated = set()  # Set to store correlated feature pairs

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            # Add the correlated feature pair to the set
            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]
            highly_correlated.add((feature_i, feature_j))

# Removing one feature from each highly correlated pair
for feature_pair in highly_correlated:
    feature_i, feature_j = feature_pair
    
    if feature_j in encoded_df.columns:
        encoded_df.drop(feature_j, axis=1, inplace=True)
        # print(encoded_df)
        
        # handling Nan values in the encoded_dataset
imputer = SimpleImputer(strategy='mean')
encoded_df_imputed = imputer.fit_transform(encoded_df)
        
        
# Applying SelectKBest to select k best features based on mutual information
selector = SelectKBest(score_func=mutual_info_classif, k=200)

new_dataset = selector.fit_transform(encoded_df_imputed,target )

# Getting the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Getting the names of the selected features
selected_features = encoded_df.columns[selected_feature_indices]


# Filtering the dataset based on selected features
selected_features = list(selected_features)

# Creating a new DataFrame for filtered data
df_filtered = pd.DataFrame()
df_filtered['target_variable'] = dataset['target_variable']  # Copy the target variable column

# Iterate through selected features and copy the corresponding columns
for feature in selected_features:
    df_filtered[feature] = df[feature]

# Remove rows with missing target variable values
df_filtered.dropna(subset=['target_variable'], inplace=True)




# print("Selected features:")
# for feature in selected_features:
#     print(feature)

# Using Xgboost Algorithm to train our model

# Loading the Iris dataset
iris = load_iris()
new_dataset, target = iris.data, iris.target

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(new_dataset, target, test_size=0.2, random_state=42)
print(target.shape)
print(encoded_df.shape)
# Converting the data into DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Setting the parameters for the XGBoost classifier
params = {
    'objective': 'binary:logistic',
    'num_class': 2,
    'n_jobs': multiprocessing.cpu_count()
}

# Training the XGBoost classifier
model = xgb.train(params, dtrain)

# Making predictions on the test set
y_pred = model.predict(dtest)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")






