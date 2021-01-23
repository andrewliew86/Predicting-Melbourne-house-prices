# Getting data from domain.com for housing price prediction Melbourne (2+ bedrooms but below 1 million)

from urllib.request import urlopen as uReq
import urllib.request
from bs4 import BeautifulSoup as soup
import re
import pandas as pd
from tqdm import tqdm # gives you a progress bar as you download stuff in a for loop
import numpy as np
import time

# Create list to store data
price_list = []
address1_list = []
address2_list = []
room_list = []
shower_list = []
car_list = []
size_list = []


# Scrape data from pages 2 - 50 (range2,51)
for i in tqdm(range(2,51)):
    # sleep is used to make sure that I dont spam the server too much
    time.sleep(2)
    try:
        my_url = "https://www.domain.com.au/sale/melbourne-region-vic/apartment/?bedrooms=2-any&price=0-1000000&excludeunderoffer=1&page=2"
        req = urllib.request.Request(my_url,headers={'User-Agent': "Magic Browser"})
        con = uReq(req)
        page_html = con.read()
        con.close()
        # html parsing
        page_soup = soup(page_html, 'html.parser')
        containers = page_soup.find_all(class_="css-qrqvvg")
        for container in containers:
            # Get price
            try:
                price_container = container.find_all('p', class_="css-mgq8yx")
                price = price_container[0].text.strip().encode('ascii', 'ignore').decode("utf-8")
                price_list.append(price)
                print(price)
            except IndexError:
                print('None')
                price_list.append('NG')
            
            # Get address
            address_container1 = container.find_all('span', class_="css-iqrvhs")
            # Address line 1
            try:
                address1_full = address_container1[0].text.strip().encode('ascii', 'ignore').decode("utf-8")
                address1_list.append(address1_full)
            except IndexError:
                print('None')
                address1_list.append('NG')
            # Address line 2
            try:
                address2_full = address_container1[1].text.strip().encode('ascii', 'ignore').decode("utf-8")
                address2_list.append(address2_full)
                print(address2_full)
            except IndexError:
                print('None')
                address2_list.append('NG')
            
            # Rooms, showers, car spaces and size of houses are all in a single container
            att_container = container.find_all('span', class_="css-1rzse3v")
            # Number of rooms
            try:
                rooms = att_container[0].text.strip().encode('ascii', 'ignore').decode("utf-8")
                print(rooms)
                room_list.append(rooms)
            except IndexError:
                print('None')
                room_list.append('NG')
                
            # Number of showers
            try:
                shower = att_container[1].text.strip().encode('ascii', 'ignore').decode("utf-8")
                print(shower)
                shower_list.append(shower)
            except IndexError:
                print('None')
                shower_list.append('NG')
            
            # Number of car spaces
            try:
                car = att_container[2].text.strip().encode('ascii', 'ignore').decode("utf-8")
                print(car)
                car_list.append(car)
            except IndexError:
                print('None')
                car_list.append('NG')
                
            # Size of house
            try:
                size = att_container[3].text.strip().encode('ascii', 'ignore').decode("utf-8")
                print(size)
                size_list. append(size)
            except IndexError:
                print('None')
                size_list.append('NG')
    except :
        continue  # In the case where there is a HTTP error or something...


# Make a dataframe.. Note that all the lists have to be the same length to create the dataframe!!
df = pd.DataFrame(np.column_stack([price_list, address1_list, address2_list, 
                                   room_list, shower_list, 
                                   car_list, size_list]),
                                   columns=['Price', 'Address1', 
                                             'Address2', 'Room', 
                                             'Shower', 'Car', 
                                             'Size'])
# df['Distance'] = df['Postcode'].map(distance_dict) # Get distance of suburbs from CBD
df.to_pickle('melb_real_estate_buy_data_23Jan21') # I pickle the dataframe so that I dont have to keep scraping the website to look at the data
print (df) # just print out the dataframe to have a look!
df = pd.read_pickle('melb_real_estate_buy_data_23Jan21')
df.to_csv('melb_real_estate_buy_data_23Jan21.csv', encoding='utf-8', index=False)