# ML models testing and prediction
# Lets start building some models to see if we can get accurate predictions
import pandas as pd 
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
X_train, y_train, X_test, y_test = train_test_split(X, y, 0.2, stratify=True)




#%%

# Use lazy predict to test multiple models 
# Lazy predict will give you an idea of which models perform the best
# YOu can then select the model and perform hyperparamter tuning to fine tune the model further
# Seee here: https://towardsdatascience.com/how-to-run-30-machine-learning-models-with-2-lines-of-code-d0f94a537e52


#%%