# Here I clean up the data that I scraped from the website and prepare the data for a machine learning model
import pandas as pd
import numpy as np
import folium  # folium is used for creating a map of the properties
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim


df = pd.read_csv('melb_real_estate_carnegie_surrounding_30Jan21.csv')
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
df['Address_folium'] = df['Address1'].str.extract(r'(\D+,)')
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











#%%
col_val_count = ['Address2', 'Shower', 'Car', 'Size']

# Return the proportion of 'unique values' of the 'Address2', 'Shower', 'Car', 'Size' columns in the dataframe
for d in col_val_count:
    print(df[d].value_counts(normalize=True))

# About 5.7% of properties are from the Melbourne Vic 3000 postcode so most of the data is spread out across various suburbs
# About 58% of the properties have only 1 bath and ~42% have two baths
# 78% of properties have 1 parking spot. Note that 1? property did not show any value for parking spot
# Unfortunately, 91% of properties do not have floor size information