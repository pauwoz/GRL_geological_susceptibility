#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for maps visualisation.
"""

import os # if want to change path

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, StrMethodFormatter, MultipleLocator
from shapely import geometry

os.chdir('D:/GEOL_SUSC')

deb_tops = pd.read_csv("figures/manuscript/April_30.04/input_params_to_visualize/debolt_tops.csv", delimiter=',')
mon_tops = pd.read_csv("figures/manuscript/April_30.04/input_params_to_visualize/montney_tops.csv", delimiter=',')
mon_thick = pd.read_csv("figures/manuscript/April_30.04/input_params_to_visualize/montney_thickness.csv", delimiter=',')
prec_tops = pd.read_csv("figures/manuscript/April_30.04/input_params_to_visualize/precambr_tops.csv", delimiter=',')
azi_var = pd.read_csv("figures/manuscript/April_30.04/input_params_to_visualize/azi_montney.csv", delimiter=',')
pressure = pd.read_csv("figures/manuscript/April_30.04/input_params_to_visualize/pressure.csv", delimiter=',')
dist_db = pd.read_csv("data/FINAL/final_inputs/dist_belt_distances_grid.csv", header=0, delimiter=',')
dist_maj_faults = pd.read_csv("data/FINAL/final_inputs/major_faults_grid.csv", header=0, delimiter=',')

montney_poly = pd.read_csv('shape_files/montney_polygon_coords.csv')

coords_mon = np.array(list(zip(montney_poly.lon, montney_poly.lat)))
poly = geometry.Polygon(coords_mon)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(7,8), sharex=True, sharey=True)

dist_db_gdf = gpd.GeoDataFrame(
    dist_db, geometry=gpd.points_from_xy(dist_db.field_1, dist_db.field_2))
dist_db_mon = dist_db_gdf[dist_db_gdf.within(poly)]

plt.subplots_adjust(hspace=0.3, wspace=0.15)
for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8):
    ax.set_ylim(52, 60)
    ax.set_xlim(-124.4, -115.3)
    ax.set_yticklabels(np.arange(52, 60, 1))
    ax.set_xticklabels(np.arange(-125, -115, 1), rotation=0)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.grid(c='grey', alpha=0.5)
    ax.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°N"))
    ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°W"))

ax1.set_title('Montney top depth [m]', fontsize=9)
im = ax1.scatter(mon_tops['lon'], mon_tops['lat'], c=mon_tops['mon_top'], s=0.1, vmin=0, vmax=5500, cmap='inferno')
cbar=plt.colorbar(im,ax=ax1)

ax2.set_title('Montney thickness [m]', fontsize=9)
im = ax2.scatter(mon_thick['lon'], mon_thick['lat'], c=mon_thick['mon_thick'], s=0.1)
cbar=plt.colorbar(im,ax=ax2)

ax3.set_title('Debolt top depth [m]', fontsize=9)
im = ax3.scatter(deb_tops.lon, deb_tops.lat, c=deb_tops['deb_top'],s=0.1, vmin=0, vmax=5500, cmap='inferno')
cbar = plt.colorbar(im,ax=ax3)

ax4.set_title('Precambrian top depth [m]', fontsize=9)
im = ax4.scatter(prec_tops.lon, prec_tops.lat, c=prec_tops['prec_top'], s=0.1,vmin=0, vmax = 5500, cmap='inferno')
cbar = plt.colorbar(im,ax=ax4)

ax5.set_title('Pressure gradient [kPa/m]', fontsize=9)
im = ax5.scatter(pressure.lon, pressure.lat, c=pressure['press_grad'],cmap='cividis', s=0.1)
cbar=plt.colorbar(im,ax=ax5)
cbar.set_ticks([5,10,15])
cbar.set_ticklabels([5,10,15])

ax6.set_title('SHmax azimuth variance from 45\u00b0 [\u00b0]', fontsize=9)
im = ax6.scatter(azi_var.lon,azi_var.lat, c=azi_var['azi'],cmap='Greens', s=0.1, vmin=-10)
cbar = plt.colorbar(im,ax=ax6)

ax7.set_title('Proximity to Cordilleran belt [km]', fontsize=9)
im = ax7.scatter(dist_db_mon.field_1, dist_db_mon.field_2, c=dist_db_mon['distance'], cmap='binary_r', s=0.1)
ax7.contour(dist_db_mon[['field_1', 'field_2']].values)
cbar = plt.colorbar(im,ax=ax7)
cbar.set_ticks([50000,150000,250000])
cbar.set_ticklabels([50,150,250])

ax8.set_title('Proximity to faults [km]', fontsize=9)
im = ax8.scatter(dist_maj_faults.field_1, dist_maj_faults.field_2, c=dist_maj_faults['distance'],cmap='binary_r', s=0.1)
ax4.contour(dist_maj_faults[['field_1', 'field_2']].values)
cbar = plt.colorbar(im,ax=ax8)
cbar.set_ticks([50000,100000,150000])
cbar.set_ticklabels([50,100,150])

plt.tight_layout()
plt.show()
# plt.savefig("input_parameters.png", dpi=300)
