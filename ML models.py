# ML models testing and prediction
# Lets start building some models to see if we can get accurate predictions
import pandas as pd 
import numpy as np
from matplotlib import pyplot
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor

# Get the 'clean' housing data (output from the 'data cleaning and EDA' file)
df = pd.read_csv("house_price_prediction.csv")
# I've noticed that all the houses 

# Label encode the suburb column
# Create instance of labelencoder
labelencoder = LabelEncoder()
# Replace the suburb column with encoded data
df['Suburb'] = labelencoder.fit_transform(df['Suburb'])

# There are missing values in the Price column, I could try and use KMeans imputation but I am keeping it easy and going with just removing the NaNs
df.dropna(subset=['Final_price'], inplace=True)

# Assert (test) that there are no longer any NaNs in our dataset
assert df.isna().sum().sum() < 1

#%%
# Lets randomly sample the dataframe to keep 10% of the data (I dont have alot of data) for final testing of optimized model
# NOTE: DO not touch the test_only_set until we are ready to test the final model!
train_opt_set = df.sample(frac=0.90,random_state=200) #random state is a seed value
test_only_set = df.drop(train_opt_set.index)


# Prepare dataset for testing by splitting the dataset into 80% train and then 20% test
X = train_opt_set.drop("Final_price", axis=1)
y = train_opt_set["Final_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)




#%%
# Before performing ML, determine the naive price prediction 
# Calculate the performance (RMSE)  when mean or median of train dataset are used to naively predict the price of houses 
rmse_naive_mean = (sum((y_test - np.mean(y_train))**2)/len(y_test))**(1/2)
rmse_naive_med = (sum((y_test - np.median(y_train))**2)/len(y_test))**(1/2)
# Using the median price of the training set, we get an RMSE of 77448
# Any machine learning model RMSE should be lower than this to be worth pursuing! 

#%%
# Lazy predict will give you an idea of which models perform the best - it tests multiple models to give you an idea of which one is best!
# You can then select the model and perform hyperparamter tuning to fine tune the model further
# See here: https://towardsdatascience.com/how-to-run-30-machine-learning-models-with-2-lines-of-code-d0f94a537e52
# Note: If you have problems installing lazypredict (because of xgboost), install xgboost first (seperately) using conda and then try installing lazypredict 
# Fit all models
reg = LazyRegressor(predictions=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Interesting! The top 5 models (when sorted by R-squared and RMSE) are all variations of linear regression! 
# HuberRegressor was found to be the best base model with an RMSE of 61979.8 - which is lower than the rmse_naive_med
# HuberRegressor is a linear regression technique that is more 'tolerant' to outliers which could be one reason why the other models performed poorly
# The HuberRegressor is different to Ridge because it applies a linear loss to samples that are classified as outliers. A sample is classified as an inlier if the absolute error of that sample is less than a certain threshold.
# See here for explanation of Huber regression https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
#%%
# Another idea to imporve performance is to stack the different regression models
# See https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/

# Get a stacking ensemble of models
# Here we are stacking 4 models (KNN, huber regressor, gradient boosting and then linear regression) 
def get_stacking():
    """ Models used in stacking regressor"""
    level0 = list()
    level0.append(('knn', KNeighborsRegressor()))
    level0.append(('huber', HuberRegressor()))
    level0.append(('gbr', GradientBoostingRegressor()))
    # define meta learner model
    level1 = LinearRegression()
    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=3)
    return model

# Get a list of models to evaluate
# This is so that we can compare the performance of individual models vs stacking them together
def get_models():
    """Dictionary containing individual models used in the stack (for eval purposes)"""
    models = dict()
    models['knn'] = KNeighborsRegressor()
    models['huber'] = HuberRegressor()
    models['gbr'] = GradientBoostingRegressor()
    models['stacking'] = get_stacking()
    return models

# Evaluate a given model using repeated 3 fold cross-validation
def evaluate_model(model, X, y):
    """Evaluates each model using Kfold validation and calculates the RMSE for each model"""
    cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# Get the models to evaluate
models = get_models()
# Evaluate the models and store results
results, names = list(), list()

print("Negative root mean squared error for each model (+/- standard deviation):")
# Use a for loop to evaulate all the models
for name, model in models.items():
    scores = evaluate_model(model, X_train, y_train)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    
# Plot model performance for comparison using a boxplot
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

# Stacking did not seam to improve the model hugely compared with Huber although Huber does a have a few outliers in its RMSE values
# Further hyperparameter tuning with the huber regresor is probably the best option