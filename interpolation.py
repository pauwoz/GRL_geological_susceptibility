#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:49:45 2019

@author: paulina
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from matplotlib.ticker import StrMethodFormatter
import utm
import geopandas as gpd

os.chdir('D:/GEOL_SUSC/')
#
df = pd.read_csv("data/draft/Atlas_well_control_data_WCSB.csv", header=0, delimiter=',')
df = df.loc[df['prcs']>0]

df_gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lon, df.lat))

df['easting']=0
df['northing']=0

# recalculate coordinates from lat/lon to northing/easting

zone = 11
for i in df.index:

    df.loc[i, 'easting'] = list(utm.from_latlon(df.loc[i, 'lat'], df.loc[i, 'lon'], zone, 'U'))[0]  # easting
    df.loc[i, 'northing'] = list(utm.from_latlon(df.loc[i, 'lat'], df.loc[i, 'lon'], zone, 'U'))[1]  # northing
    df.loc[i, 'zone'] = list(utm.from_latlon(df.loc[i, 'lat'], df.loc[i, 'lon'], zone, 'U'))[2]  # zone
    df.loc[i, 'zone_letter'] = list(utm.from_latlon(df.loc[i, 'lat'], df.loc[i, 'lon'], zone, 'U'))[3]  # zone letter

df = df[['prcs', 'lon', 'lat', 'easting', 'northing']]

df.rename(columns={'easting': 'x',  # depending on the format of input data 
                   'northing': 'y'}, inplace=True) 

step = 2500
extent = x_min, x_max, y_min, y_max = [100161.17256633355, 598195.012411898, 5839614.174655632, 6654300.485022815]
grid_x, grid_y = np.mgrid[x_min:x_max:step, y_min:y_max:step]

rbfi = Rbf(df.x, df.y, df.prcs, function='linear', smooth=200000)
di = rbfi(grid_x, grid_y)

import utm

coords = zip(grid_x.flatten(), grid_y.flatten())
df_interp = pd.DataFrame(coords, columns=['easting', 'northing'])

# recalculate interpolated values with utm coordinates back in lat/lon
zone = 11
for c in df_interp.index:
    df_interp.at[c, 'lat'] = list(utm.to_latlon(df_interp.at[c, 'easting'], df_interp.at[c, 'northing'], zone, 'U'))[0]
    df_interp.at[c, 'lat'] = list(utm.to_latlon(df_interp.at[c, 'easting'], df_interp.at[c, 'northing'], zone, 'U'))[1]

mapped = zip(df_interp.lon, df_interp.lat, df_interp.easting, df_interp.northing, di.flatten())
mapped = set(mapped)
df_2 = pd.DataFrame(mapped, columns=['lon', 'lat', 'easting', 'northing', 'prcs'])

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely import geometry

montney_poly = pd.read_csv("shape_files/montney_polygon_coords.csv")
coords = np.array(list(zip(montney_poly.lon, montney_poly.lat)))
xs, ys = zip(*coords)  # create lists of x and y values
coords = list(zip(montney_poly.lon, montney_poly.lat))


fig, ax = plt.subplots()
ax.legend('Pressure gradient (kPa/m)')
ax.grid(c='grey')


# adding disturbed belt
dist_belt = pd.read_csv("shape_files/dist_belts_nodes_utm11.csv")
ax.plot(dist_belt.xcoord, dist_belt.ycoord, c='k', linestyle='-', label='Disturbed Belt')

ax.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f} °N"))
ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f} °W"))

im = ax.imshow(di.T, extent=(min(df_2.lon), max(df_2.lon), min(df_2.lat), max(df_2.lat)),
               origin='lower', cmap='viridis', clip_path=patch, clip_on=True)

fig.colorbar(im, ax=ax, orientation='horizontal')

montney_poly = pd.read_csv('shape_files/montney_polygon_coords.csv')

coords_mon = np.array(list(zip(montney_poly.lon, montney_poly.lat)))
poly = geometry.Polygon(coords_mon)
susc_gdf = gpd.GeoDataFrame(
    df_2, geometry=gpd.points_from_xy(df_2.lon, df_2.lat))
susc_gdf_montney = susc_gdf[susc_gdf.within(poly)]
print("Minimal value within Montney:", susc_gdf_montney.prcs.max())
print("Maximum value within Montney:", susc_gdf_montney.prcs.min())

# np.savetxt('data/FINAL/interpolation_results/interpolated_grid.csv', di, delimiter=',') 
# df_2.to_csv('data/FINAL/interpolation_results/interpolated_table .csv')


# plt.scatter(df_2.lon, df_2.lat, c=df_2.prcs,s=10)
# plt.show()
