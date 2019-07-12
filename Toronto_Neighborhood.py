#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Import necessary libraries

from bs4 import BeautifulSoup

import requests

import pandas as pd
import numpy as np

import csv

## Request the html source from web-address

source=requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M").text

#source.status_code # to check returned text is html or not

## Parse the source using BeautifulSoup

soup=BeautifulSoup(source, 'lxml')

## find the table tag in html

table=soup.find('table')
table

## Create csv file, write headers in it, find all 'tr' tags and then find all 'td' tags and store them in list and write list into file. Last close the file

with open('Toronto.csv','w') as csv_file:
    csv_writer=csv.writer(csv_file)
    csv_writer.writerow(['PostalCode','Borough','Neighborhood'])
    for tb in table.find_all('tr'):
        list1=[]
        for head in tb.find_all('td'):
            list1.append(head.text)
        csv_writer.writerow(list1)

csv_file.close()

## Read csv file, drop rows with Borough=Not assigned

df=pd.read_csv('Toronto.csv')

df=df[df.Borough!='Not assigned']

df.head(10)

df1=df.reset_index(drop=True) # reset index to 0

## Replace the unnecessary characters from Neighborhood column

df1.Neighborhood=df1.Neighborhood.str.replace("\\r\n","") 
df1.head(10)

## Set Neighborhood=Borough where Neighborhood is 'Not assigned'

df1['Neighborhood'].loc[df1['Neighborhood']=='Not assigned']=df1['Borough']
df1.head(10)

df1.shape # find shape

## Combine the rows with same PostalCode

res = df1.groupby(['PostalCode','Borough'])['Neighborhood'].agg( ','.join).reset_index()

res

res.shape #check the shape


# In[8]:


df2=pd.read_csv('http://cocl.us/Geospatial_data/Geospatial_Coordinates.csv')

df2.head()

res1=pd.concat([res,df2],axis=1)

res1.head()


# In[ ]:





# In[13]:


print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(res1['Borough'].unique()),
        res1['Neighborhood'].shape
    )
)


# In[16]:


from geopy.geocoders import Nominatim 
address = 'Toronto, On'

geolocator = Nominatim(user_agent="on_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of New York City are {}, {}.'.format(latitude, longitude))


# ## Create a map of Toronto with neighborhoods superimposed on top.

# In[17]:


# create map of New York using latitude and longitude values
import folium
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(res1['Latitude'], res1['Longitude'], res1['Borough'], res1['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# In[47]:


#toronto_data = res1[res1.Borough.replace({'Toronto':np.nan},regex=True).isnull()]
downtown_data = res1[res1['Borough'] == 'Downtown Toronto'].reset_index(drop=True)
downtown_data.Neighborhood=downtown_data.Neighborhood.str.replace("\\n","") 
downtown_data


# In[39]:


address = 'Downtown Toronto, ON'

geolocator = Nominatim(user_agent="on_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Downtown Toronto are {}, {}.'.format(latitude, longitude))


# In[42]:


# create map of Manhattan using latitude and longitude values
map_downtown = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(downtown_data['Latitude'], downtown_data['Longitude'], downtown_data['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_downtown)  
    
map_downtown


# In[43]:


CLIENT_ID = 'OC0GPD4H3HWLICOX3Y243SO4XMFAUFSV2E5GR4USPUPUSBZE' # your Foursquare ID
CLIENT_SECRET = 'OHV5JOGVGHWQAQQVRCA303CL2YQYVV2QLYLIFEN0JNXP3RTG' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[48]:


downtown_data.loc[0, 'Neighborhood']


# In[49]:


neighborhood_latitude = downtown_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = downtown_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = downtown_data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# In[50]:


#Double-click __here__ for the solution.
#<!-- The correct answer is:
LIMIT = 100  # limit of number of venues returned by Foursquare API

#<!--
radius = 500 # define radius
#-->
#<!--
# create URL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL
#--> 


# In[51]:


results = requests.get(url).json()
results


# In[52]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[57]:


from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[55]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# In[58]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[60]:


downtown_venues = getNearbyVenues(names=downtown_data['Neighborhood'],
                                   latitudes=downtown_data['Latitude'],
                                   longitudes=downtown_data['Longitude']
                                  )


# In[62]:


print(downtown_venues.shape)
downtown_venues.head()


# In[63]:


downtown_venues.groupby('Neighborhood').count()


# In[65]:


print('There are {} uniques categories.'.format(len(downtown_venues['Venue Category'].unique())))


# ## 3. Analyse each neighborhood

# In[66]:


# one hot encoding
downtown_onehot = pd.get_dummies(downtown_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
downtown_onehot['Neighborhood'] = downtown_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [downtown_onehot.columns[-1]] + list(downtown_onehot.columns[:-1])
downtown_onehot = downtown_onehot[fixed_columns]

downtown_onehot.head()


# In[67]:


downtown_onehot.shape


# In[68]:


downtown_grouped = downtown_onehot.groupby('Neighborhood').mean().reset_index()
downtown_grouped


# In[69]:


downtown_grouped.shape


# In[70]:


num_top_venues = 5

for hood in downtown_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = downtown_grouped[downtown_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[71]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[73]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = downtown_grouped['Neighborhood']

for ind in np.arange(downtown_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(downtown_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## 4. Cluster Neighborhood

# In[75]:


# import k-means from clustering stage
from sklearn.cluster import KMeans

# set number of clusters
kclusters = 5

downtown_grouped_clustering = downtown_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(downtown_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[78]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Label', kmeans.labels_)

downtown_merged = downtown_data

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
downtown_merged = downtown_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

downtown_merged.head() # check the last columns!


# In[81]:


# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(downtown_merged['Latitude'], downtown_merged['Longitude'], downtown_merged['Neighborhood'], downtown_merged['Cluster Label']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[ ]:




