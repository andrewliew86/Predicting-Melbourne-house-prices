# Here I clean up the data that I scraped from the website and prepare the data for a machine learning model
import numpy as np
import pandas as pd

df = pd.read_csv('melb_real_estate_buy_data_23Jan21.csv')
#%%
# The first thing is to take a look at the price column as this is the most important (target column) 
print(df['Price'].value_counts(normalize=True))
# About 13% of the prices are listed as 'contact agent'. There are also strings like 'FOR SALE' in the price column. Check the proportion of these 
df['Price'] = df['Price'].astype('str')

# The first problem is that some prices are listed as '$300k'. Change these to 300,000. 
# i.e change capturing group 2 from 'k' or 'K' to ',000'
df['Price'] = df['Price'].str.replace(r'(\d+)([Kk])', r'\1,000')

# Second, there is a mixture of '-' and 'to' when indicating the price range. Change these to 'to'
df['Price'] = df['Price'].str.replace(r'\s*-\s*', r' to ')

# Third, extract all prices from the 'Price' column using a findall with regex and clean 'Price_standz_list' column using a custom function
df['Price_standz_list'] = df['Price'].str.findall(r'(\$*\d{2,3},*\d{2,3}\s*)(?:to\s*\$*\d{2,3},*\d{2,3})??')


def clean_price(price):
    """
    Removes '$', strips whitespace, calculates the mean price if a range was specified and returns nan for empty lists
    """
    if price == []:
        return np.nan
    else:
        # Note that I changed the individual prices to 'int' after stripping other characters
        clean_price = [int(i.strip().replace('$','').replace(',','')) for i in price]
        return np.mean(clean_price)
        
    
# Apply 'clean_price' function to the 'Price_standz_list' column      
df['Final_price'] = df['Price_standz_list'].apply(clean_price)

# There are still nans in our price range but it would be good to now perform some EDA

#%%
# The distance of a suburb to the CBD will likely play a role in determining the price of the property
# Use the website http://house.speakingsame.com/profile.php?q=CAULFIELD+NORTH&sta=vic to calculate distance to CBD
# First change all the 'NG' data in the 'Address2' column to take on the values of the 'Address1' column because I noticed that suburb data is present in the 'Address1' column when 'Address2' column is 'NG
df['Address2'] = np.where(df['Address2'] == 'NG', df['Address1'], df['Address2'])






#%%
col_val_count = ['Address2', 'Shower', 'Car', 'Size']

# Return the proportion of 'unique values' of the 'Address2', 'Shower', 'Car', 'Size' columns in the dataframe
for d in col_val_count:
    print(df[d].value_counts(normalize=True))

# About 5.7% of properties are from the Melbourne Vic 3000 postcode so most of the data is spread out across various suburbs
# About 58% of the properties have only 1 bath and ~42% have two baths
# 78% of properties have 1 parking spot. Note that 1? property did not show any value for parking spot
# Unfortunately, 91% of properties do not have floor size information