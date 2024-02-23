import geopandas as gpd

from shapely.geometry import Point, LineString
from . read_inputs import input_directory

def nearest_source(changing_parameter,shape_file_name,run_start_time,directory):

    suffix = '_' + run_start_time[-8:-3] + '.tif'
    suffix_with_param = '_' + run_start_time[-8:] + '_' + str(changing_parameter) + '.tif'

    potential_areas = gpd.read_file(directory + shape_file_name)
    rivers = gpd.read_file(input_directory + 'Sources\\' + 'Vienna_rivers.shp')
    rivers = rivers.to_crs(potential_areas.crs)
    rivers = rivers.loc[:,['OBJECTID','LENGTH_KM','DIS_AV_CMS','geometry']]

    nearest_line_ids = []  # List to store the IDs of the nearest lines
    nearest_distances = []  # List to store the distances between clusters and their nearest lines

    for cluster_body in potential_areas.itertuples():
        nearest_distance = float("inf")
        nearest_line_id = None

        # Create a shapely Point representing the centroid of the cluster body polygon
        centroid = cluster_body.geometry.centroid

        # Iterate through each line to find the nearest one
        for line in rivers.itertuples():
            distance = centroid.distance(line.geometry)

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_line_id = line.Index

        nearest_line_ids.append(nearest_line_id)
        nearest_distances.append(nearest_distance)
    potential_areas["nearest_river_id"] = nearest_line_ids
    potential_areas["nearest_distance"] = nearest_distances

    # Optionally, if you want to merge the information about the nearest line into the cluster bodies DataFrame:
    nearest_river = potential_areas.merge(rivers, left_on="nearest_river_id", right_index=True)

    output_csv_name = 'source_matching_' + str(changing_parameter) + '.csv'

    nearest_river.to_csv(directory + output_csv_name)
    return None