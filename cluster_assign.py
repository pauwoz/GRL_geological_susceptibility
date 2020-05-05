"""
Script includes HDBSCAN algorithm and categorizes the wells according to the conditions:
1) well has an event that occurred up to 90 days after the end of the injection
2) event is located < 5km from the well
"""
import os
import pandas as pd
import geopy.distance
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan
import utm
from sklearn.cluster import DBSCAN, KMeans
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries

import matplotlib.pyplot as plt
import time

start = time.time()

os.chdir('D:/GEOL_SUSC')

poly = pd.read_csv('shape_files/montney_polygon_coords.csv')

wells_params = pd.read_csv('WELLS_INPUT.csv', delimiter=',')  # file with parameters to merge with classification results
wells_prod_dates = pd.read_csv('WELLS_PROD_DATES.csv', delimiter=',', usecols=['UWI','First_Prod_Date','First_Prod_Time','Last_Prod_Date','Last_Prod_Time'])  # info about the production dates
wells = wells_params.merge(wells_prod_dates, on='UWI')

wells['full_date'] = wells["First_Prod_Date"] + [' '] + wells["First_Prod_Time"]
wells['full_date'] = pd.to_datetime(wells.full_date, format='%Y-%m-%d')
wells_gdf = gpd.GeoDataFrame(
    wells, geometry=gpd.points_from_xy(wells.lon, wells.lat))

event = pd.read_csv('data/EQ_catalogues/compiled_catalogue_montney.csv', skiprows=0, header=0)

event['easting'] = 0
event['northing'] = 0

zone = 10
for i in event.index:
    if (event.at[i, 'lat'] > 0) & (event.at[i, 'lon'] != 'None'):
        event.at[i, 'easting'] = list(utm.from_latlon(event.at[i, 'lat'], event.at[i, 'lon'], zone, 'U'))[0]  # easting
        event.at[i, 'northing'] = list(utm.from_latlon(event.at[i, 'lat'], event.at[i, 'lon'], zone, 'U'))[1]  # northing
        event.at[i, 'zone'] = list(utm.from_latlon(event.at[i, 'lat'], event.at[i, 'lon'], zone, 'U'))[2]  # zone
        event.at[i, 'zone_letter'] = list(utm.from_latlon(event.at[i, 'lat'], event.at[i, 'lon'], zone, 'U'))[3]  # zone letter

event['full_date'] = pd.to_datetime(event.full_date, format='%Y-%m-%d %H:%M:%S.%f')

gdf_events_xy = gpd.GeoDataFrame(
    event, geometry=gpd.points_from_xy(event.easting, event.northing))

gdf_events = gpd.GeoDataFrame(
    event, geometry=gpd.points_from_xy(event.lon, event.lat))

x = gdf_events['easting']
y = gdf_events['northing']
event['data_feature'] = gdf_events['full_date'] - gdf_events['full_date'].min()
event['data_feature'] = event['data_feature'].dt.days
X = gdf_events[['easting', 'northing', 'data_feature']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

min_cluster_size = 3

clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
clusterer.fit(X_scaled)

d = {'full_date': gdf_events.full_date, 'Magnitude': gdf_events.mag, 'lon': gdf_events.lon, 'lat': gdf_events.lat,
     'easting': gdf_events.easting, 'northing': gdf_events.northing, 'label': clusterer.labels_}
clusters_df = pd.DataFrame(data=d)

print("\nRESULTS FOR {} EVENTS IN THE CLUSTER".format(min_cluster_size))
print("Number of clustered events:", clusters_df[clusters_df['label'] != -1].shape[0])
print("Number of unclustered events:", clusters_df[clusters_df['label'] == -1].shape[0])
print("Number of clusters:", clusters_df.groupby(by='label')['label'].count().shape[0])

clusters_only = clusters_df[clusters_df['label'] != -1]
events_montney = clusters_only[clusters_only['lon'] < -120]
print("no of clusters in montney only", len(events_montney['label'].unique()))

clusters_gdf = gpd.GeoDataFrame(
    clusters_only, geometry=gpd.points_from_xy(clusters_only.lon, clusters_only.lat))

# we only assign events from the British Columbia as only they are related to Montney production
clusters_gdf = clusters_gdf[clusters_gdf['lon'] < -120]

print("Biggest cluster:", clusters_only.groupby(by='label')['label'].count().max())
print("Smallest cluster:", clusters_only.groupby(by='label')['label'].count().min())

wells_gdf['seismogenic'] = 0
wells_gdf['assigned_event'] = 0
wells_gdf['IS_event_distance'] = 0

start = time.time()

for row in wells_gdf.itertuples():
    if row.seismogenic == 0:
        for row2 in clusters_gdf.itertuples():
            coords_1 = row.lat, row.lon
            coords_1 = tuple(coords_1)
            coords_2 = row2.lat, row2.lon
            coords_2 = tuple(coords_2)

            dist = geopy.distance.distance(coords_1, coords_2).m

            if dist < 5000:
                delay_days = (row.full_date - row2.full_date) / np.timedelta64(1, 'D')
                if (delay_days > 0) & (delay_days < 90):

                    wells_gdf.at[row[0], 'seismogenic'] = 1
                    wells_gdf.at[row[0], 'assigned_event'] = row.full_date
                    wells_gdf.at[row[0], 'IS_event_distance'] = dist

                    break
                else:
                    continue
                break
            else:
                continue
end = time.time()

print("number of seismogenic wells:", wells_gdf[wells_gdf.seismogenic == 1].shape[0])

fig, ax1 = plt.subplots(1, 1)
ax1.scatter(clusters_only.lon, clusters_only.lat, c=clusters_only.label, s=1)

# clusters_only.to_csv('induced_seismicity_clusters.csv')
# wells_gdf.to_csv('data/draft/SEISMOGENIC_WELLS.csv')
