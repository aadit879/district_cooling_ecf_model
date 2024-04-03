import os

import numpy as np
import pandas as pd

from rasterio.plot import show

import shutil

# from initialize import Parameters as Param

from . scripts.read_inputs import read_raster
from . scripts import anchor_1, clustering, read_inputs, save_results_with_param, save_results_normal, polygonize, \
    estimated_supply_sizing, result_summary
#from scripts.distance_identification import nearest_source

# from all_test_codes import new_clustering_test

from . initialize import Parameters




# from SES_results import all_plots


class MainCalculation:
    '''
    Outputs of the preliminary_calculation.anchor_assumption
    '''

    def __init__(self, parameters_instance):
        for attr_name, attr_value in vars(parameters_instance).items():
            setattr(self, attr_name, attr_value)

        self.output_log = self.output_directory + save_results_normal.current_time + '\\' + 'info.log'
        #self.log_parameters()

    # def log_parameters(self):
    #     logging_text = "\n\n"
    #     params_dict = vars(self)
    #     col_width = 5 + max(len(key) for key in params_dict.keys())
    #     for key in params_dict.keys():
    #         value = params_dict[key]
    #         if not isinstance(value, np.ndarray) or len(value.shape) != 2:
    #             logging_text = logging_text + "".join(key.ljust(col_width) + str(value).ljust(col_width) + '\n')
    #     with open(self.output_log, 'w') as f:
    #         f.write(logging_text)

    # Input Variables
    def all_runs(self,output_raster_demand_covered,output_raster_levl_grid_cost, output_raster_network_length,
                 output_raster_grid_investment_cost,output_raster_average_diameter, output_shp):

        electircity_price_EurpKWh = self.electricity_prices
        print('Electricity Price: ', electircity_price_EurpKWh)
        ## calculating the grid expansion costs for the distribution grid per MWh for all cells
        levl_dist_grid_cost_per_mwh = anchor_1.Levl_dist_grid(self.interest_rate, self.depreciation_dc,
                                                              self.grid_investment_costs_Eur,
                                                              self.non_ind_areas_mwh)  ## both anchors and neighbours

        save_results_normal.write_tiff(levl_dist_grid_cost_per_mwh,  self.gt, 'levl_dist_grid_cost_per_mwh',
                                       self.output_directory)

        ## final outputs
        potential_demand_MWh = self.N_ued_mwh.copy()
        potential_demand_MWh = anchor_1.zero_to_nan(potential_demand_MWh)
        potenital_demand_MW = self.N_peak_MW.copy()
        # potenital_demand_MW = anchor_1.zero_to_nan(potenital_demand_MW)

        potential_anchors_MWh = self.PA_II.copy()
        potential_anchors_MWh = anchor_1.zero_to_nan(potential_anchors_MWh)
        potential_anchors_MW = self.PA_II_MW.copy()
        potential_anchors_MW = anchor_1.zero_to_nan(potential_anchors_MW)

        #######################################################################################################################
        files_save_list = ['potential_demand_MWh', 'potenital_demand_MW', 'potential_anchors_MWh',
                           'potential_anchors_MW']

        run_start_time = save_results_normal.current_time
        output_directory = self.output_directory + '/' + run_start_time + '/'
        for prim_name in files_save_list:
            file_name = prim_name + '_' + run_start_time[-8:-3] + '.tif'
            if not os.path.exists(output_directory + file_name):
                save_results_normal.write_tiff(vars()[prim_name], self.gt, prim_name, self.output_directory)
        #######################################################################################################################
        # LCOC_threshold = changing_parameter
        ## clustering based on the identified in samples
        # anchor_df = read_raster('potential_anchors_MW' + '_' + save_results_normal.current_time[-8:-3] + '.tif',
        #                         input_directory=read_inputs.input_directory2)[3]
        anchor_df = potential_anchors_MW
        anchor_df = clustering.anchor_points(anchor_df)  ## converts the anchor points to binary

        function_run = Parameters.individaul_LCOC_anchors(self)

        LCOC_ind_cap_op = function_run[0]  # LCOC only in regions where no cooling supply exists

        LCOC_capop_anchors = LCOC_ind_cap_op / anchor_df * anchor_df  ## only anchors but indicidual System before clusters are identified
        LCOC_capop_anchors[np.isnan(LCOC_capop_anchors)] = 0

        average_LCOC_ind_anchors = LCOC_capop_anchors[LCOC_capop_anchors != 0].mean()  ## average based only on anchors
        LCOC_threshold = self.individual_threshold_cap * average_LCOC_ind_anchors  ## threshold defined by the average of the LCOC_anchors
        print('##########')
        print(average_LCOC_ind_anchors)
        print(LCOC_threshold)

        ################################################################################################################
        clusters = clustering.identify_potential_clusters(levl_dist_grid_cost_per_mwh, self.aued_mwh, self.pipe_length,
                                                          anchor_df, LCOC_threshold,
                                                          self.output_directory)  ## both anchors and neighbours

        clusters_numbered, cluster_count = polygonize.label_clusters(clusters)

        cluster_mask = clusters != 0
        show(clusters)

        changing_parameter = round(LCOC_threshold)

        anchor_levelized_cost = np.zeros(self.PA_II.shape)
        anchor_cells = np.where(self.PA_II != 0)
        anchor_levelized_cost[anchor_cells] = levl_dist_grid_cost_per_mwh[anchor_cells]

        # save_results_with_param.write_tiff(anchor_levelized_cost, 'anchor_levelized_cost', changing_parameter,
        #                                    self.output_directory)

        print('Unique elements in cluster:' + str(np.unique(clusters)))
        print('Total clusters (num):' + str(len(np.unique(clusters_numbered))))
        print('Total cluster mask cells :' + str(cluster_mask.sum()))

        if not np.any(clusters > 0):
            print(
                'No clusters identified for electricity price of ' + str(electircity_price_EurpKWh * 1000) + ' Eur/MWh')

            clusters, anchor_df, Average_levl_dist_grid_cost_per_mwh, network_length, anchor_to_cluster, \
            LCOC_threshold, tot_demand_expan, avg_LCOC_ind_clusters, tot_inv_grid, tot_gfa,cluster_shape, symbol_list1 = np.zeros(
                12)

            sensitivity = pd.DataFrame()
            sensitivity.loc[0, 'Theoretical Cooling Demand (TWh)'] = round(self.tued_mwh.sum() / 1000000, 2)
            sensitivity.loc[0, 'Actual Cooling Demand (TWh)'] = round(self.aued_mwh.sum() / 1000000, 2)

            return (
                clusters, anchor_df, Average_levl_dist_grid_cost_per_mwh,
                network_length, anchor_to_cluster, LCOC_threshold, tot_demand_expan,
                avg_LCOC_ind_clusters, tot_inv_grid, tot_gfa,
                sensitivity, cluster_shape, symbol_list1)


        print('if statment bypassed!!')

        clusters = anchor_1.zero_to_nan(clusters)

        #######################################################################################################################

        #######################################################################################################################
        levl_dist_grid_cost_per_mwh = anchor_1.zero_to_nan(levl_dist_grid_cost_per_mwh)
        # save_results_with_param.write_tiff(levl_dist_grid_cost_per_mwh, 'levl_dist_grid_cost_per_mwh',
        #                                    changing_parameter,self.output_directory)

        # os.rename(output_directory + 'LCOC_ind_all_'+ save_results_normal.current_time[-8:-3] + '.tif',
        #           output_directory + 'LCOC_ind_all_' + save_results_normal.current_time[-8:-3] + '_' + str(
        #               changing_parameter) + '.tif')

        # LCOC_ind = anchor_1.zero_to_nan(LCOC_ind)
        # save_results.write_tiff(LCOC_ind, 'LCOC_ind', changing_parameter)

        # save_results_with_param.write_tiff(self.grid_investment_unit_costs_Eurperm,
        #                                    'grid_investment_unit_costs_Eurperm', changing_parameter,
        #                                    self.output_directory)

        save_results_with_param.write_tiff(clusters, self.gt, 'cluster_default', changing_parameter, self.output_directory)
        print('Anchor_df_sum: ' + str(anchor_df.sum()))
        anchor_df = anchor_1.zero_to_nan(anchor_df)
        print('Anchor_df_sum: ' + str(anchor_df.sum()))
        # save_results_with_param.write_tiff(anchor_df, 'potential_anchors_default', changing_parameter,
        #                                    self.output_directory)

        ## cluster cut-outs
        # demand_covered = potential_demand_MWh / clusters * clusters

        # Average_levl_dist_grid_cost_per_mwh = levl_dist_grid_cost_per_mwh / clusters * clusters  # both anchors and neighbours
        Average_levl_dist_grid_cost_per_mwh = np.where(cluster_mask, levl_dist_grid_cost_per_mwh, 0)
        print('Average_levl_dist_grid_cost_per_mwh:' + str(Average_levl_dist_grid_cost_per_mwh.mean()))
        Average_levl_dist_grid_cost_per_mwh[np.isnan(Average_levl_dist_grid_cost_per_mwh)] = 0
        print('Average_levl_dist_grid_cost_per_mwh:' + str(Average_levl_dist_grid_cost_per_mwh.mean()))
        save_results_with_param.write_tiff(Average_levl_dist_grid_cost_per_mwh, self.gt, 'Average_levl_dist_grid_cost_per_mwh',
                                           changing_parameter, self.output_directory)

        # Average_pipe_diameter = self.diameter_mm / clusters * clusters  # both anchors and neighbours
        Average_pipe_diameter = np.where(cluster_mask, self.diameter_mm, 0)
        print('Max pipe diameter:' + str(Average_pipe_diameter.max()))
        print('Min pipe diameter:' + str(Average_pipe_diameter.min()))
        Average_pipe_diameter[np.isnan(Average_pipe_diameter)] = 0
        print('Max pipe diameter:' + str(Average_pipe_diameter.max()))
        print('Min pipe diameter:' + str(Average_pipe_diameter.min()))

        # Total_grid_investment = self.grid_investment_costs_Eur / clusters * clusters  # both anchors and neighbours
        Total_grid_investment = np.where(cluster_mask, self.grid_investment_costs_Eur, 0)
        print('Total_grid_investment:' + str(Total_grid_investment.sum()))
        Total_grid_investment[np.isnan(Total_grid_investment)] = 0
        print('Total_grid_investment:' + str(Total_grid_investment.sum()))

        # LCOC_ind_anchors = LCOC_capop_anchors / clusters * clusters  ## LCOC of individual system (AC) for anchors selected in the cluster
        LCOC_ind_anchors = np.where(cluster_mask, LCOC_capop_anchors, 0)
        print('LCOC_ind_anchors:' + str(LCOC_ind_anchors.mean()))
        LCOC_ind_anchors[np.isnan(LCOC_ind_anchors)] = 0
        print('LCOC_ind_anchors:' + str(LCOC_ind_anchors.mean()))
        avg_LCOC_ind_anchors = round(LCOC_ind_anchors[LCOC_ind_anchors != 0].mean(), 2)
        print('avg_LCOC_ind_anchors:' + str(avg_LCOC_ind_anchors))

        # LCOC_ind_clusters = LCOC_ind_cap_op / clusters * clusters  ## LCOC of individual system (AC) for anchors + neighbours
        LCOC_ind_clusters = np.where(cluster_mask, LCOC_ind_cap_op, 0)
        print('LCOC_ind_clusters:' + str(LCOC_ind_clusters.mean()))
        LCOC_ind_clusters[np.isnan(LCOC_ind_clusters)] = 0
        print('LCOC_ind_clusters:' + str(LCOC_ind_clusters.mean()))
        save_results_with_param.write_tiff(LCOC_ind_clusters, self.gt, 'LCOC_ind_clusters', changing_parameter,
                                           self.output_directory)
        avg_LCOC_ind_clusters = round(LCOC_ind_clusters[LCOC_ind_clusters != 0].mean(), 2)
        print('avg_LCOC_ind_clusters:' + str(avg_LCOC_ind_clusters))

        ## network length based on the road
        # network_length = self.pipe_length / clusters * clusters
        network_length = np.where(cluster_mask, self.pipe_length, 0)
        print('network_length:' + str(network_length.sum()))
        print('network_length_mean:' + str(network_length.mean()))
        network_length[np.isnan(network_length)] = 0
        print('network_length:' + str(network_length.sum()))
        print('network_length_mean:' + str(network_length.mean()))
        save_results_with_param.write_tiff(network_length, self.gt, 'network_length', changing_parameter, self.output_directory)

        #######################################################################################################################
        # anchor_to_cluster = anchor_df / clusters * clusters
        print('###Anchor DF: ' + str(anchor_df.max()))
        print('###Anchor DF: ' + str(anchor_df.min()))
        anchor_to_cluster = np.where(cluster_mask, anchor_df, 0)
        print('anchor_to_cluster:' + str(anchor_to_cluster.sum()))
        anchor_to_cluster = anchor_1.zero_to_nan(anchor_to_cluster)
        print('anchor_to_cluster:' + str(anchor_to_cluster.sum()))
        save_results_with_param.write_tiff(anchor_to_cluster, self.gt, 'anchor_to_cluster', changing_parameter,
                                           self.output_directory)


        anchor_to_cluster[np.isnan(anchor_to_cluster)] = 0
        print('anchor_to_cluster_sum:' + str(anchor_to_cluster.sum()))


        anchor_points_count = anchor_to_cluster.sum()
        # number of anchors that show actual feasibility for conversion into DC network central points


        avg_levl_dist_grid_cost_per_mwh_anchor, tot_demand,\
        per_demand, avg_levl_dist_grid_cost_per_mwh_expan,\
        tot_demand_expan, per_demand_expan, tot_inv_grid,\
            tot_gfa,symbol_list1 = result_summary.print_summary(
            clusters, anchor_points_count, potential_demand_MWh, cluster_mask, changing_parameter,
            Average_levl_dist_grid_cost_per_mwh, anchor_to_cluster, avg_LCOC_ind_anchors, Total_grid_investment, self)
        ####################################################################################

        changed_parameter = LCOC_threshold
        changed_parameter_type = 'Threshold'
        # columns = ['Electricity_Price', changed_parameter_type, 'Average_LCOC', '% Demand_Covered',
        #            '% Demand_Covered_Expansion', 'Average_LCOC_Expansion', 'Average_LCOC_ind_all_capop',
        #            'Average_LCOC_ind_anchors', 'Feasible_anchors', '%rise_with_expansion']

        sensitivity = result_summary.summary_df_generator(electircity_price_EurpKWh, changed_parameter,
                            changed_parameter_type, anchor_df, avg_levl_dist_grid_cost_per_mwh_anchor,
                            tot_demand, per_demand, avg_levl_dist_grid_cost_per_mwh_expan,
                            tot_demand_expan, per_demand_expan, tot_inv_grid, tot_gfa, anchor_points_count,
                            avg_LCOC_ind_anchors, avg_LCOC_ind_clusters, self)

        directory = self.output_directory
        csv_name = 'sensitivity_default_run' + str(self.csv_file_number) + '.csv'

        if not os.path.exists(directory + csv_name):
            sensitivity.to_csv(directory + csv_name, index=False)
        else:
            dict = sensitivity.to_dict('records')
            df = pd.read_csv(directory + csv_name)
            df = df.append(dict, ignore_index=True)

            df.to_csv(directory + csv_name, index=False)
        polygonize_raster_name = 'cluster_default_' + run_start_time[-8:] + '_' + str(changing_parameter) + '.tif'

        # output_polygon_name = 'polygon_' + run_start_time[-8:] + '_' + str(changing_parameter) + '.shp'
        # polygonize.polygonize(polygonize_raster_name, output_directory, output_polygon_name)

        # polygonize.cluster_parameters_allocation(changing_parameter, output_polygon_name, run_start_time,
        #                                          output_directory, electircity_price_EurpKWh)

        # estimated_supply_sizing.supply_sizing(self.COP_DC, self.flh_cooling_days, self.af_dc, self.cf_dc,
        #                                        self.interest_rate, self.depreciation_dc, output_polygon_name,
        #                                        output_directory, electircity_price_EurpKWh)

        ##############################################################################################################
        # save_results_normal.write_tiff(cluster_mask, 'cluster_mask', CM.output_directory)

        save_results_normal.write_tiff(clusters_numbered, self.gt, 'cluster_numbered', self.output_directory)

        cluster_numbered_name = 'cluster_numbered_' + save_results_normal.current_time[-8:-3] + '.tif'
        ######################################
        #files in the var/temp directory
        # print(cluster_numbered_name)
        #
        # print(self.output_directory)
        #
        # all_entries = os.listdir(self.output_directory)
        # print(all_entries)
        # # Filter out directories, leaving only files
        # files = [entry for entry in all_entries if os.path.isfile(os.path.join(directory, entry))]
        # print(files)
        #
        # temporary_directory = os.path.join(self.output_directory, run_start_time)
        # print(os.listdir(temporary_directory))




        ##########################################
        output_polygon_name = output_shp


        polygonize.polygonize(cluster_numbered_name, output_directory, output_polygon_name)

        polygonize.cluster_parameters_allocation(changing_parameter, output_polygon_name, run_start_time,
                                                 output_directory, electircity_price_EurpKWh)

        ##############################################################################################################

        # cluster_polygon_name = 'cluster_shape_' + run_start_time[-8:-3] + '_' + str(changing_parameter) + '.shp'
        # polygonize.cluster_parameters_allocation(changing_parameter, cluster_polygon_name, run_start_time,
        #                                          output_directory, electircity_price_EurpKWh)

        # assessment of connection to the nearest free cooling supply sources

        cluster_shape = estimated_supply_sizing.supply_sizing(self.COP_DC, self.flh_cooling_days, self.af_dc, self.cf_dc,
                                              self.interest_rate, self.depreciation_dc, output_polygon_name,
                                              output_directory, electircity_price_EurpKWh)

        # nearest_source(changing_parameter, output_polygon_name, run_start_time, output_directory)

        # nearest_source(changing_parameter, cluster_polygon_name, run_start_time, output_directory)
        # all_plots.plotter()

        demand_covered = np.where(cluster_mask, self.aued_mwh, 0)
        save_results_normal.write_tiff(demand_covered, self.gt, output_raster_demand_covered, self.output_directory,
                                       current_time_bool = False)

        levl_grid_cost = np.where(cluster_mask, Average_levl_dist_grid_cost_per_mwh, 0)
        save_results_normal.write_tiff(levl_grid_cost, self.gt, output_raster_levl_grid_cost, self.output_directory,
                                       current_time_bool = False)

        save_results_normal.write_tiff(network_length, self.gt, output_raster_network_length,
                                           self.output_directory,current_time_bool = False)

        save_results_normal.write_tiff(Total_grid_investment, self.gt, output_raster_grid_investment_cost, self.output_directory,
                                        current_time_bool = False)

        save_results_normal.write_tiff(Average_pipe_diameter, self.gt, output_raster_average_diameter, self.output_directory,
                                       current_time_bool=False)


        # check if the directory exists
        if os.path.exists(os.path.join(self.output_directory + '/' + run_start_time)):
            #delete the directory
            shutil.rmtree(os.path.join(self.output_directory + '/' + run_start_time))



        print('#######################################################################################################')

        return (
            clusters, anchor_df, Average_levl_dist_grid_cost_per_mwh,
            network_length, anchor_to_cluster, LCOC_threshold,tot_demand_expan,
            avg_LCOC_ind_clusters, tot_inv_grid, tot_gfa,
            sensitivity, cluster_shape, symbol_list1)

