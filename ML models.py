# ML models testing and prediction
# Lets start building some models to see if we can get accurate predictions
import pandas as pd 
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

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
# The HuberRegressor is different to Ridge because it applies a linear loss to samples that are classified as outliers. A sample is classified as an inlier if the absolute error of that sample is lesser than a certain threshold.
# See here for explanation of Huber regression https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
#%%
# Now, perform hyperparamater tuning with Huber regressor 