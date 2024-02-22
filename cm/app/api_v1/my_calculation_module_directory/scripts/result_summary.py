import numpy as np
import pandas as pd

from scripts import polygonize
from scripts import save_results_with_param


def print_summary(clusters, anchor_points_count, potential_demand_MWh, cluster_mask, changing_parameter,
                  Average_levl_dist_grid_cost_per_mwh,anchor_to_cluster, avg_LCOC_ind_anchors,
                  Total_grid_investment, self):

    print('Total number of feasible anchor points : ' + str(anchor_points_count))
    ####################################################################################
    # demand_met_by_anchors = potential_demand_MWh / anchor_to_cluster * anchor_to_cluster
    demand_met_by_anchors = np.where(cluster_mask ,potential_demand_MWh ,0)
    demand_met_by_anchors[np.isnan(demand_met_by_anchors)] = 0
    tot_demand = round(demand_met_by_anchors.sum(), 2)
    per_demand = round(demand_met_by_anchors.sum() / self.aued_mwh.sum() * 100, 2)
    print('Total met demand (MWh) : ' + str(tot_demand))
    print('% of theoretical demand met : ' + str(per_demand))

    # demand_met_by_expansion = potential_demand_MWh / clusters * clusters
    demand_met_by_expansion = np.where(cluster_mask ,potential_demand_MWh ,0)
    demand_met_by_expansion[np.isnan(demand_met_by_expansion)] = 0
    symbol_list1 = polygonize.symbol_list_creation(demand_met_by_expansion)

    save_results_with_param.write_tiff(demand_met_by_expansion, 'Demand_met_by_expansion',
                                       changing_parameter ,self.output_directory)
    tot_demand_expan = round(demand_met_by_expansion.sum(), 2)
    per_demand_expan = round(demand_met_by_expansion.sum() / self.aued_mwh.sum() * 100, 2)
    ## this also includes the anchors
    print('Total met demand feasible locations (MWh) : ' + str(tot_demand_expan)  )  # anchors + demand
    print('% of theoretical demand met : ' + str(per_demand_expan))

    ## percentage rise with the clusters compared to only anchor length
    per_rise_with_cluster = (tot_demand_expan - tot_demand) / tot_demand * 100
    print('Increase in the demand coverage with expansion :' + str(round(per_rise_with_cluster, 2)) + '%')
    ####################################################################################

    levl_dist_grid_cost_per_mwh_of_anchors = Average_levl_dist_grid_cost_per_mwh / anchor_to_cluster * anchor_to_cluster  ## distribution_grid_cost of anchors in cluster
    levl_dist_grid_cost_per_mwh_of_anchors[np.isnan(levl_dist_grid_cost_per_mwh_of_anchors)] = 0
    avg_levl_dist_grid_cost_per_mwh_anchor = round(
        levl_dist_grid_cost_per_mwh_of_anchors[levl_dist_grid_cost_per_mwh_of_anchors != 0].mean(), 2)
    print('Average Distribution-Grid-cost of the selected DC anchors DC (Eur/MWh) : ' + str(
        avg_levl_dist_grid_cost_per_mwh_anchor))

    levl_dist_grid_cost_per_mwh_of_expansion = Average_levl_dist_grid_cost_per_mwh.copy() # distribution_grid_cost of anchors + neighbours
    levl_dist_grid_cost_per_mwh_of_expansion[np.isnan(levl_dist_grid_cost_per_mwh_of_expansion)] = 0
    avg_levl_dist_grid_cost_per_mwh_expan = round(
        levl_dist_grid_cost_per_mwh_of_expansion[levl_dist_grid_cost_per_mwh_of_expansion != 0].mean(), 2)
    print('Average Distribution-Grid-cost of the potential DC after expansion (Eur/MWh) : ' + str(
        avg_levl_dist_grid_cost_per_mwh_expan))

    # avg_LCOC_ind = function_run[1] # LCOC_ind for anchors + neighbours
    print('Average LCOC of individual supply for the identified Anchors : ' + str(avg_LCOC_ind_anchors))

    # avg_LCOC_ind_for_anchors = LCOC_threshold
    # print('Average LCOC of corresponding individual supply : ' + str(avg_LCOC_ind))
    ####################################################################################

    # total_investment_anchors = Total_grid_investment / anchor_to_cluster * anchor_to_cluster
    total_investment_anchors = np.where(cluster_mask ,Total_grid_investment ,0)
    total_investment_anchors[np.isnan(total_investment_anchors)] = 0
    tot_inv = round(total_investment_anchors.sum(), 2)
    print('Total grid investment on anchors (Euros): ' + str(tot_inv))
    total_investment_grid = Total_grid_investment / clusters * clusters
    total_investment_grid[np.isnan(total_investment_grid)] = 0
    tot_inv_grid = round(total_investment_grid.sum(), 2)
    print('Total investment on grid (Euros): ' + str(tot_inv_grid))
    save_results_with_param.write_tiff(total_investment_grid, 'total_investment_grid',
                                       changing_parameter ,self.output_directory)

    # cooling_gfa = self.gfa_m2 / clusters * clusters
    cooling_gfa = np.where(cluster_mask, self.gfa_m2, 0)
    cooling_gfa[np.isnan(cooling_gfa)] = 0
    save_results_with_param.write_tiff(cooling_gfa, 'cooling_gfa', changing_parameter,
                                       self.output_directory)
    tot_gfa = round(cooling_gfa.sum(), 2)
    print('Total covered cooling GFA (m2): ' + str(tot_gfa))

    return (avg_levl_dist_grid_cost_per_mwh_anchor, tot_demand,per_demand,\
            avg_levl_dist_grid_cost_per_mwh_expan,\
            tot_demand_expan,per_demand_expan,\
            tot_inv_grid, tot_gfa,symbol_list1 )

def summary_df_generator(electircity_price_EurpKWh, changed_parameter,
                            changed_parameter_type, anchor_df, avg_levl_dist_grid_cost_per_mwh_anchor,
                            tot_demand, per_demand, avg_levl_dist_grid_cost_per_mwh_expan,
                            tot_demand_expan, per_demand_expan, tot_inv_grid, tot_gfa, anchor_points_count,
                            avg_LCOC_ind_anchors, avg_LCOC_ind_clusters, self):


    sensitivity = pd.DataFrame()
    sensitivity.loc[0, 'Electricity_Price'] = electircity_price_EurpKWh
    sensitivity.loc[0, changed_parameter_type] = changed_parameter

    sensitivity.loc[0, 'Residential Share'] = self.res_dist_factor
    sensitivity.loc[0, 'Non-residential Share'] = self.nonres_dist_factor
    sensitivity.loc[0, 'Potential Anchors'] = len(anchor_df[anchor_df == 1])  # anchors that could be feasible
    sensitivity.loc[
        0, 'Average_levl_dist_grid_anchor (Eur/MWh)'] = avg_levl_dist_grid_cost_per_mwh_anchor  ## LCOC DC of only anchor points
    sensitivity.loc[0, 'Demand_Covered by anchors (MWh)'] = tot_demand
    sensitivity.loc[0, '% Demand_Covered by anchors'] = per_demand

    sensitivity.loc[
        0, 'Average_levl_dist_grid_neighbour (Eur/MWh)'] = avg_levl_dist_grid_cost_per_mwh_expan  ## Overall average LCOC DC of the entire grid (anchor + neighbour)
    sensitivity.loc[0, 'Demand_Covered by expansion (MWh)'] = tot_demand_expan
    sensitivity.loc[0, '% Demand_Covered_after_Expansion'] = per_demand_expan

    sensitivity.loc[
        0, 'Average_corresponding_LCOC_ind_of_anchors (Eur/MWh)'] = avg_LCOC_ind_anchors  #### avg_LCOC_ind # all the possible grids excluding the assumed 30%
    sensitivity.loc[
        0, 'Average_corresponding_LCOC_ind_of_clusters (Eur/MWh)'] = avg_LCOC_ind_clusters  # only the anchors

    sensitivity.loc[
        0, 'Feasible_anchors'] = anchor_points_count  # anchors that are feasible based on the calculations
    # sensitivity.loc[0, '%rise_with_expansion'] = per_rise_with_cluster
    # sensitivity.loc[0, 'Total Investment only anchors (Euros)'] = tot_inv
    sensitivity.loc[0, 'Total Investment Grid (Euros)'] = tot_inv_grid  # anchors and neighbours
    sensitivity.loc[0, 'Total floor area covered in m2'] = tot_gfa

    sensitivity.loc[0, 'Theoretical Cooling Demand (TWh)'] = self.tued_mwh.sum() / 1000000
    sensitivity.loc[0, 'Actual Cooling Demand (TWh)'] = self.aued_mwh.sum() / 1000000
    sensitivity.loc[0, 'DC Covered Demand (TWh)'] = tot_demand_expan / 1000000

    return sensitivity