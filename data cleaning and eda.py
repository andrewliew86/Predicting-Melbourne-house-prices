# Here I clean up the data that I scraped from the website (on two seperate occassions to get more data) and prepare the data for a machine learning model
import pandas as pd
import numpy as np
import folium  # folium is used for creating a map of the properties
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas.api.types as ptypes


# Read in the data that was scraped on two seperate occasions and combine into single dataframe
df1 = pd.read_csv('melb_real_estate_carnegie_surrounding_30Jan21.csv')
df2 = pd.read_csv('melb_real_estate_carnegie_surrounding_28March21.csv')
df = pd.concat([df1, df2], axis=0)

# Remove duplicates (defined as having the same price and address1)
df = df.drop_duplicates(subset=['Price', 'Address1'], keep='last')

#%%
# The first thing is to take a look at the price column as this is the most important (target column) 
print(df['Price'].value_counts(normalize=True))
# Some prices are listed as 'contact agent'. There are also strings like 'FOR SALE' in the price column.
# Clearning of price is needed so convert column to string first for cleaning
df['Price'] = df['Price'].astype('str')

# The first problem is that some prices are listed as '$300k'. Change these to 300,000. 
# i.e change capturing group 2 from 'k' or 'K' to ',000'
df['Price'] = df['Price'].str.replace(r'(\d+)([Kk])', r'\1,000')

# Second, there might be a mixture of '-' and 'to' when indicating the price range. Change these to 'to'
df['Price'] = df['Price'].str.replace(r'\s*-\s*', r' to ')

# Third, extract all prices from the 'Price' column using a findall with regex and clean 'Price_standz_list' column using a regex
df['Price_standz_list'] = df['Price'].str.findall(r'(\$*\d{2,3},*\s*\d{2,3}\s*)(?:to\s*\$*\d{2,3},*\s*\d{2,3})??')

# Finally, clean up the price with a custom function
def clean_price(price):
    """
    Removes '$', strips whitespace, calculates the mean price if a range was specified and returns nan when price is not specified
    """
    if price == []:
        return np.nan
    else:
        # Note that I changed the individual prices to 'int' after stripping other characters
        clean_price = [int(i.strip().replace('$','').replace(',','').replace(' ','')) for i in price]
        return np.mean(clean_price)
        
    
# Apply 'clean_price' function to the 'Price_standz_list' column      
df['Final_price'] = df['Price_standz_list'].apply(clean_price)

# There are still nans in our price range but it would be good to now perform some EDA

#%% Draw a Folium map to check that all the addresses are in the same proximity of each other 
# Extract the street name and add combine street name with suburb name for the geolcator 
df['Address_folium'] = df['Address1'].str.extract(r'(\D+,)');
df['Address_folium'] = df['Address_folium'] + ' ' + df['Address2']

# Note, the following folium and geolocator code was adapted from here: https://towardsdatascience.com/pythons-geocoding-convert-a-list-of-addresses-into-a-map-f522ef513fd6
# Use geolocator to get latitude and longitude data from street address
geolocator = Nominatim(user_agent="my-test-app")
df["loc"] = df["Address_folium"].apply(lambda x: geolocator.geocode(x) if pd.notnull(x) else None)
df["point"]= df["loc"].apply(lambda loc: tuple(loc.point) if loc else None)
df[['lat', 'lon', 'altitude']] = pd.DataFrame(df['point'].to_list(), index=df.index)

# Create Folium map
# Convert loc column to string first 
df['loc'] = df['loc'].map(str)  
# Create a new dataframe and remove rows that are NaN for the address (geolocator has mistakenly return this data as an address in Italia - see 'Nanno') 
folium_df = df.loc[~df['loc'].str.contains('Nanno', na=False), :]
# Create a map object and center it to the avarage coordinates to m
m = folium.Map(location=folium_df[["lat", "lon"]].mean().to_list(), zoom_start=5)
# If the points are too close to each other, cluster them, create a cluster overlay with MarkerCluster, add to m
marker_cluster = MarkerCluster().add_to(m)
# Draw the markers and assign popup and hover texts
# Add the markers the the cluster layers so that they are automatically clustered
for i,r in folium_df.iterrows():
    location = (r["lat"], r["lon"])
    folium.Marker(location=location,
                      popup = r['Final_price'],
                      tooltip=r['Final_price']).add_to(marker_cluster)
# Save the map
m.save("folium_map_melb_prop_31Jan21.html")
# Open the HTML to look at the interactive map. Hover over each property to see the price. Price that is not available is indicated by 'nan'
# We see that most of the datapoints are approximately in the correct suburbs

#%%
# The distance of a suburb to the CBD will likely play a role in determining the price of the property
# Use the website http://house.speakingsame.com/profile.php?q=CAULFIELD+NORTH&sta=vic to calculate distance to CBD

# First change all the 'NG' data in the 'Address2' column to take on the values of the 'Address1' column because I noticed that suburb data is present in the 'Address1' column when 'Address2' column is 'NG
df['Address2'] = np.where(df['Address2'] == 'NG', df['Address1'], df['Address2'])

# Get suburb data from Address2 column
df['Suburb'] = df['Address2'].str.extract(r'(\D+)\d* VIC')
df['Suburb'] = df['Suburb'].str.replace(' ', '+')

# Create a new column that indicates the distance from the city based on the dictionary of suburb-price pairs 
dist_dict = {'caulfield':9.3,
'elsternwick':8.6,
'gardenvale':10,
'glen+huntly':10.8,
'mckinnon':12.6,
'murrumbeena':12.5,
'ormond':12.1,
'carnegie':11.8,
'bentleigh':13.2,
'bentleigh+east':14.2}

df['dist_cbd'] = df['Suburb'].str.lower().map(dist_dict)

#%%
# Some more tidying up before exploratory data analysis
# First, remove the columns that won't be used
df.drop(columns=['Price', 'Address1', 'Price_standz_list'], inplace=True)

# Count unique 
col_val_count = ['Suburb', 'Room', 'Shower', 'Car', 'Size']
# Return the proportion of 'unique values' of the 'Suburb', 'Shower', 'Car', 'Size' columns in the dataframe
for d in col_val_count:
    print(df[d].value_counts(normalize=True))

# Determine the number of missing values in each column
percent_missing = df.isnull().sum() * 100 / len(df)
print(percent_missing.sort_values(ascending=False))

# About 47% of properties are from CARNEGIE so this suburb will be the most accurate (likely) for predictions
# About 57% of the properties have 2 baths and 42% have one bath
# 95% of properties have 1 parking spot. 
# Unfortunately, 88% of properties do not have floor size information so this data is likely to be less useful for house price predictions!
# With Nans, it appears that ~22% of the price data is missing which is bad but is not unexpected!


#%%
# Now, perform some exploratory data analysis
# What is the distribution of house prices?
df['Final_price'].hist()
plt.title('Distribution of house prices')
plt.show()
# Distibution of house prices appears to be right skewed and median house price is between 550000 and 600000

# Does the location affect house prices?
sns.boxplot(x='Final_price', y='Suburb', data=df)
plt.title('House prices grouped by suburb')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# It looks like Claufield might have the highest median price and cMckinnon might have the lowest. 
# The range for Bentleigh east and Glen Huntley appears to be the largest

# Does number of bedrooms, carparks and showers affect price?
cols_to_plot = ['Room', 'Shower', 'Car']
for i in cols_to_plot:
    sns.catplot(x=i, y='Final_price', hue='Suburb', data=df)
    plt.title(i)
    plt.show()
# The class imbalance caused by the lack of data for each column (class) makes it difficut to interpret but generally the higher the number of bedrooms and baths, tend to be more expensive and this is independent of location!

# What about distance to CBD and price of house? 
g = sns.JointGrid(x="Final_price", y="dist_cbd", data=df)
g.plot_joint(sns.scatterplot, s=100, alpha=.5)
g.plot_marginals(sns.kdeplot, shade =True)
g.annotate(stats.pearsonr)  # annote with the pearson correlation
plt.show()    
# There doesnt seem to be much of any correlation but a better measure could be to use the exact location of the house (based on address) and calculate distance to the CBD
#%%
# Now, prep the data for machine learning 
# For 'Room', 'Shower' and 'Car' columns, there are string characters (e.g. 2 rooms) so we need to keep numeric characters only
df["Room"] = df["Room"].str.extract('(\d+)')
df["Shower"] = df["Shower"].str.extract('(\d+)')
df["Car"] = df["Car"].str.extract('(\d+)')

# Use assert to make sure all strings have been removed from those columns (Check all columns are numeric)
cols_to_check = ["Room", "Shower", "Car"]
assert all(ptypes.is_numeric_dtype(df[col]) for col in cols_to_check)

# Lets remove columns that will not be used for ML modelling
df_final = df.drop(["Address2", "Size"], axis=1)

# Export data to a csv file.
# index=False is used to prevent pandas from creating the index column when exporting dataframe to csv
df_final.to_csv("house_price_prediction.csv", index=False)
    
    
    