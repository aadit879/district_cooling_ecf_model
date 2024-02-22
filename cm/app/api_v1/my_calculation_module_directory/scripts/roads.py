import pandas as pd
import geopandas as gpd

grid = gpd.read_file('G:\\My Drive\\TU WIEN\\Work\\PhD\\M1\\Input Data\\grid_vectorized_vienna.geojson')
road = gpd.read_file('G:\\My Drive\\TU WIEN\\Work\\PhD\\M1\\Input Data\\OSM_Data\\roads_vienna.geojson')
road = road.to_crs('EPSG:3035')

cells_with_road = grid.iloc[:, :].clip(road)
cells_with_road['length_m'] = cells_with_road.geometry.length
new_grid = pd.merge(grid, cells_with_road[['id', 'length_m']], on='id', how='outer')
new_grid['length_m'] = new_grid['length_m'].fillna(0)

new_grid.to_file('G:\\My Drive\\TU WIEN\\Work\\PhD\\M1\\Input Data\\new_grid.geojson')
new_grid.to_file('G:\\My Drive\\TU WIEN\\Work\\PhD\\M1\\Input Data\\roads_grid_aggregated.geojson')


import osmnx as ox
import matplotlib.pyplot as plt

place_name = "Kamppi, Helsinki, Finland"

graph = ox.graph_from_place(place_name)
