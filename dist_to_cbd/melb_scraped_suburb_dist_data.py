# The distance of a suburb to the CBD will likely play a role in determining the price of the property
# Use the website http://house.speakingsame.com/profile.php?q=CAULFIELD+NORTH&sta=vic to calculate distance to CBD

from urllib.request import urlopen as uReq
import urllib.request
from bs4 import BeautifulSoup as soup
import re
import pandas as pd
from tqdm import tqdm # gives you a progress bar as you download stuff in a for loop
import numpy as np
import time


# Get a list of all the suburbs in our dataset
suburb = set(df['Suburb'].values)
distance_list = []

# # Scrape data from website to get distance from CBD to suburb and then place data into a csv called 'distance_to_cbd.csv'
for i in tqdm(suburb):
    try:
        # sleep is used to make sure that I dont spam the server too much
        time.sleep(15)
        my_url = "http://house.speakingsame.com/profile.php?q={}&sta=vic".format(i)
        req = urllib.request.Request(my_url,headers={'User-Agent': "Magic Browser"})
        con = uReq(req)
        page_html = con.read()
        con.close()
        # html parsing and getting the distance from cbd text
        page_soup = soup(page_html, 'html.parser')
        dist_text = page_soup.select('#mainT table table td+ td a')[0].text.strip().encode('ascii', 'ignore').decode("utf-8")
        distance = re.findall(r'(\d{1,2}\.*\d)km', dist_text)[0]
        distance_list.append(distance)
    except Exception as e: 
        # print error if it occurs
        print(str(e))
        distance_list.append('NG')

# Create a csv file containing the distance to the CBD using scraped data
df_distance = pd.DataFrame(np.column_stack([list(suburb), distance_list]), columns=['Suburb', 'Distance'])    
df_distance.to_csv('distance_to_cbd.csv', encoding='utf-8', index=False)
