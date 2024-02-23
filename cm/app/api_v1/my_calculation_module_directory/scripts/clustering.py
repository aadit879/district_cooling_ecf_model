import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd

from scipy.spatial import cKDTree

from . polygonize import polygonize
from . save_results_normal import write_tiff, current_time

import warnings

warnings.filterwarnings('ignore')

global_clusters = set()
coords = None
tree = None
anchor_indices = None



## function to extract anchor to binary
# PA_II_MW
def anchor_points(identified_locations):
    '''
    :param identified_locations:
    :return: cnonverts the identified anchors into binary raster
    '''
    # indentified_location = anchor_1
    potential_anchors = identified_locations.copy()
    potential_anchors[np.isfinite(potential_anchors)] = 1
    potential_anchors[np.isnan(potential_anchors)] = 0

    return potential_anchors

def neighbor_indentifier(neighborhood_size):
    # Create a meshgrid for the neighborhood
    x_offsets, y_offsets = np.meshgrid(np.arange(-neighborhood_size, neighborhood_size + 1),
                                       np.arange(-neighborhood_size, neighborhood_size + 1))
    # Stack the x and y offsets to create the neighbor_offsets
    neighbor_offsets = np.dstack((x_offsets, y_offsets)).reshape(-1, 2)
    return neighbor_offsets


def idx_value_identfier(tree, x, y, neighbor_size):
    # previous level neghborhood to remove already considered cells
    neighbor_offsets_minus_one = neighbor_indentifier(neighbor_size - 1)
    neighbor_offsets = neighbor_indentifier(neighbor_size)
    # Calculate the indices of the neighborhood cells around the anchor
    neighbor_coordinates_minus_one = neighbor_offsets + [x, y]
    neighbor_coordinates = neighbor_offsets + [x, y]
    dist_values, idx_values = tree.query(neighbor_coordinates)
    neighborhood = idx_values[idx_values != tree.query((x, y))[1]]

    return neighborhood


def grid_cluster(raster_layer, demand, pipe_length, anchor_r, anchor_c, threshold,output_directory):
    # Create a raster with random values
    global min_coord
    global global_clusters, coords, tree, anchor_indices
    raster = raster_layer


    cluster_indices = []  # the actual final clusters
    if np.any(np.all(coords[list(global_clusters)] == [anchor_r, anchor_c], axis=1)):
        return cluster_indices
    else:
        cells = 1
        #average = raster[anchor_r, anchor_c] / cells
        average = 0

        selected_indices_mins = [
            [anchor_r, anchor_c]]  ## all the selected mins plus anchor from where extension is possible

        neighbour_indices = []  # overall list from which the minimum selection is done

        limit_size = 10
        limit_size_neighbors = idx_value_identfier(tree,anchor_r,anchor_c,limit_size)
        limit_size_neighbors = np.setdiff1d(limit_size_neighbors, np.array(list(global_clusters)))

        pipe_length_threshold = 10000 # in meters
        pipe_length_in_cluster = 0

        while average < threshold and pipe_length_in_cluster < pipe_length_threshold:

            inds = selected_indices_mins[-1]

            point = (inds[0], inds[1])
            dist, idx = tree.query(point)

            if idx not in cluster_indices:
                cluster_indices.append(idx)

            # Find the indices of all 8 cells surrounding the selected cell
            indices = idx_value_identfier(tree,point[0],point[1],1)

            # Get the coordinates of the cell with the minimum value
            indices = np.setdiff1d(indices,cluster_indices)
            neighbour_indices = np.unique(np.concatenate((neighbour_indices, indices))).astype(int)
            neighbour_indices = np.intersect1d(neighbour_indices, limit_size_neighbors)

            nearby_anchors = np.intersect1d(neighbour_indices, anchor_indices)

            if len(neighbour_indices) == 0:
                break

            elif len(nearby_anchors) > 0:
                selection_points = nearby_anchors

            else:
                selection_points = neighbour_indices


            selection_range = np.intersect1d(selection_points, neighbour_indices)

            nonzero_indices = np.nonzero(raster.flatten()[selection_range] != 0)[0]
            nonzero_values = raster.flatten()[selection_points][np.where(raster.flatten()[selection_points] != 0)]

            if len(nonzero_values) == 0:

                break
            else:
                min_index = selection_range[nonzero_indices][np.argmin(raster.flatten()[selection_range][nonzero_indices])]
                min_coord = coords[min_index]

                neighbour_indices = np.delete(neighbour_indices, np.where(neighbour_indices == min_index)[0])

                latest_min_value = raster.flatten()[min_index]

            selected_indices_mins.append([min_coord[0], min_coord[1]])

            if len(cluster_indices) == 1:
                if raster.flatten()[cluster_indices]>threshold:
                    cluster_indices.append(min_index)
                    total_levl_cost = raster.flatten()[cluster_indices] * demand.flatten()[cluster_indices]
                    average = np.sum(total_levl_cost) / np.sum(demand.flatten()[cluster_indices])

                    if average > threshold:
                        cluster_indices = []
                        break


            if len(cluster_indices) > 700:  # the last cell addition is only done in the next loop os len() + 1
                # On average 200m of pipe length per cell. Two way 400 m. Thus 25 cells to maintain a total grid length
                # of the grid under 10km
                average = threshold + 1
            else:
                cells += 1
                #weighted average
                if len(cluster_indices) == cells:
                    cluster_inidices_n_1 = cluster_indices
                else:
                    cluster_inidices_n_1 = np.append(cluster_indices, min_index) # test cluster_indices for the loop

                total_levl_cost = raster.flatten()[cluster_inidices_n_1] * demand.flatten()[cluster_inidices_n_1]
                average = np.sum(total_levl_cost) / np.sum(demand.flatten()[cluster_inidices_n_1])

                pipe_length_in_cluster = pipe_length.flatten()[cluster_inidices_n_1].sum()

                #average = ((average * (cells - 1) + latest_min_value)) / cells



        # if len(cluster_indices) > 0:
        #     independent_cluster = np.zeros(raster.shape)
        #     np.put(independent_cluster, cluster_indices, 1)
        #     write_tiff(independent_cluster, 'temp_cluster_raster', output_directory)
        #     temp_shape_file_name = 'cluster_shape_' + str(current_time[-8:-3]) + '_' + str(round(threshold)) + '.shp'
        #     if os.path.exists(output_directory + current_time + '\\' + temp_shape_file_name):
        #         polygonize('temp_cluster_raster_' + current_time[-8:-3] + '.tif',
        #                    output_directory + current_time + '\\', 'temp_shape.shp')
        #         existing_gdf = gpd.read_file(output_directory + current_time + '\\' + temp_shape_file_name)
        #         temp_gdf = gpd.read_file(output_directory + current_time + '\\' + 'temp_shape.shp')
        #         existing_gdf = gpd.GeoDataFrame(pd.concat([existing_gdf, temp_gdf], ignore_index=True),
        #                                         crs=existing_gdf.crs)
        #
        #         existing_gdf.to_file(output_directory + current_time + '\\' + temp_shape_file_name)
        #         matching_files = glob.glob(os.path.join(output_directory + current_time + '\\', 'temp_shape*'))
        #         for file_path in matching_files:
        #             os.remove(file_path)
        #     else:
        #         polygonize('temp_cluster_raster_' + current_time[-8:-3] + '.tif',
        #                    output_directory + current_time + '\\', temp_shape_file_name)
        #     os.remove(output_directory + current_time + '\\' + 'temp_cluster_raster_' + current_time[-8:-3] + '.tif')
        #
        global_clusters = global_clusters.union(set(cluster_indices) - global_clusters)

    return cluster_indices


def identify_potential_clusters(working_area_raster, demand, pipe_length, anchor_df, threshold,output_directory):
    # create an empty numpy array with the same shape as demand raster
    global global_clusters
    global coords, tree, anchor_indices

    clusters = np.zeros(working_area_raster.shape)

    potential_anchors = anchor_df
    positions = np.argwhere(potential_anchors == 1)

    if coords is None:
        # Get the coordinates of each cell in the working shape
        coords = np.array(
            np.meshgrid(np.arange(working_area_raster.shape[0]), np.arange(working_area_raster.shape[1]))).T.reshape(-1,
                                                                                                                     2)
        # Create a KDTree object with the raster coordinates
        tree = cKDTree(coords)
        anchor_indices = tree.query(positions)[1]

    def apply_grid_cluster(anchor, threshold, output_directory):
        indices = grid_cluster(working_area_raster, demand, pipe_length, anchor[0], anchor[1], threshold, output_directory)
        np.put(clusters, indices, 1)
        return clusters

    def run_clustering(positions, threshold,output_directory):
        clusters = np.apply_along_axis(lambda x: apply_grid_cluster(x, threshold,output_directory), 1, positions)
        return clusters[-1]

    clusters = run_clustering(positions, threshold,output_directory)

    global_clusters = set()

    # all cluster have value one and the rest have zero
    return clusters

