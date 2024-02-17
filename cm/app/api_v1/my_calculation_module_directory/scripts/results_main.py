from scripts import save_results_normal


def save_raster_files(cluster_mask, demand, levl_grid_cost, network_length,
                      output_raster_clusters, output_raster_demand_covered, output_raster_anchor_df,
                      output_raster_levl_grid_cost, output_raster_network_length,output_raster_anchor_to_cluster,
                      output_raster_unit_grid_investment_cost,output_raster_grid_investment_cost):

    save_results_normal