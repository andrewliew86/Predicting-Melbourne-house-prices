# The number of restaurants in each suburb might also have an effect on the price of a property'
# To get the number of restaurants, we can use foursquare API 
# Foursquare uses the latitude and longitue of a location and returns a list of venues (e.g. restaurants) within a specified radius of that location

import requests

category_id = "4d4b7105d754a06374d81259"  # food cattegory
CLIENT_ID = input("Enter Foursquare client ID: ")  # client ID from developer account
CLIENT_SECRET = input("Enter Foursquare client secret: ")  # client secret from developer account
VERSION = "20180604"
LIMIT = 1000  # Number of results limit
neighborhood_latitude = -37.88867  # latitude of location/suburb
neighborhood_longitude = 145.05713  # longitude of location/suburb
radius = 500  # what is the radius of the search from the centre of lat and long (in metres)

# QUery the foursquare API
url = "https://api.foursquare.com/v2/venues/explore?&categoryId={}&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}".format(
    category_id,
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)

# Read the returned JSON results
results = requests.get(url).json()

# Note, you will need to run the above (with different lat and long) for each suburb with a for loop

#%%
# Obtain a list of venues with the specific restaurant names
venues = results['response']['groups'][0]['items']

# Get the number of restuarants returned from the location 
resto_num = len(venues)
# Append the resto_num to each house address

