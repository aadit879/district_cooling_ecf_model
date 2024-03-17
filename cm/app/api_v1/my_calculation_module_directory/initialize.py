import os
import numpy as np
import logging

from . scripts.read_inputs import read_raster
from . scripts import read_inputs, anchor_1, individual_costs, save_results_normal


class Parameters:
    def __init__(self, electricity_prices, flh_cooling_days, COP_DC, delta_T_dc, ind_tec_SEER, interest_rate,
                 depriciation_dc, depriciation_ac, int_raster_gfa_tot, int_raster_gfa_non_res, int_raster_cdm,
                 output_directory):

        # primary user inputs
        self.electricity_prices = electricity_prices
        self.flh_cooling_days = flh_cooling_days
        self.COP_DC = COP_DC
        self.delta_T_dc = delta_T_dc
        self.ind_tec_SEER = ind_tec_SEER
        self.interest_rate = interest_rate
        self.depreciation_dc = depriciation_dc
        self.depreciation_ac = depriciation_ac

        self.gfa_m2 = read_inputs.read_raster2(int_raster_gfa_tot)[3]
        self.gfa_nonres_m2 = read_inputs.read_raster2(int_raster_gfa_non_res)[3]
        demand_rasters = read_inputs.read_raster2(int_raster_cdm)
        self.tued_mwh = demand_rasters[3]
        self.gt = demand_rasters[0]

        self.output_directory = output_directory

        # secondary user inputs; not possible to change from the web tool
        # ratio of actual covered demand to theoretical demand
        # potential service distribution factor
        self.per_actual_demand = 0.3  # Default: 0.3 in case of Vienna

        # 1. self.electricity_prices = [0.01, 0.08, 0.2,0.5]
        self.csv_file_number = 414

        self.pipe_proxy = 1  # 1: Plot_ratio; 2: Road
        if self.pipe_proxy == 2:
            print('Read road inputs')
            # self.road_m = read_raster('road_raster_grid_aggregated.tif')[3]

        # percentile for the thresholds
        self.maximum_energy_threshold = 75  # (default: 75 )
        self.minimum_gfa_threshold = 25  # (default: 25 )
        self.anchor_definition_type = 1  # 1: Areas with GFA greater than the minimum_gfa_threshold; 2: less than
        self.minimum_peak_threshold = 1  # (default: 1 MW )
        self.anchor_MW_threshold = [45, 55]  # values inbetween 55 and 45 th percentile are only taken
        # Technical parameters
        # 2. self.flh_cooling_days = 60  # (default:60 for case of Vienna )

        self.af_dc = 1
        self.cf_dc = 1
        # 3. self.COP_DC = 4.89
        # 4. self.delta_T_dc = 10  # delta T in the cooling network
        self.avg_fluid_velocity_mperS = 1.5  # velocity of fluid flow in pipes; Literature indicates a range of 1.5 - 3 m/s

        # 5. self.ind_tec_SEER = 3.6
        self.af_ind = 0.9
        self.cf_ind = 0.7
        self.individual_threshold_cap = 1  ## the individual threhsold is capped to avoid the total LCOC_DC exceeding

        # Financial Parameters
        # 6. self.interest_rate = 0.06
        # 7. self.depreciation_dc = 25
        # 8. self.depreciation_ind = 12

        self.identify_areas = True  # True: Input is all cells; Identfies area irrespective of current supply technology;  # False: Discards areas which already have supply based on self.per_actual_demand

        # Assumption of Service Distribution Factor
        self.res_dist_factor = 0.8 #0.1  # 5%
        self.nonres_dist_factor = 0.8 #0.5  # 15%

        #non-res: total ratio, > self.non_res_threhold is classified as a non_residential cell
        self.non_res_threhold = 0.2

        # self.res_dist_factor = [(0.01,0.1),(0.08,0.1),(0.01,0.2),(0.05,0.15),(0.08,0.2)]

    def working_area_identification(self):

        ## filter out high residential areas
        self.non_res_ratio = np.zeros_like(self.gfa_m2)
        cells = np.where(self.gfa_m2 != 0)
        self.non_res_ratio[cells] = self.gfa_nonres_m2[cells] / self.gfa_m2[cells]

        ## assumptions on the service distribution factor
        non_res_cells_mask = np.where(self.non_res_ratio > self.non_res_threhold)
        res_cells_mask = np.where((self.non_res_ratio <= self.non_res_threhold) & (self.non_res_ratio != 0))

        self.aued_mwh = np.zeros_like(self.tued_mwh)
        self.aued_mwh[non_res_cells_mask] = self.tued_mwh[non_res_cells_mask] * self.nonres_dist_factor
        self.aued_mwh[res_cells_mask] = self.tued_mwh[res_cells_mask] * self.res_dist_factor

        if self.identify_areas == False:
            # needs to be checked; but most likely not used in any case.
            self.individual_supplied = individual_costs.random_selector(self.aued_mwh, self.gfa_m2,
                                                                        self.per_actual_demand)
        else:
            self.individual_supplied = np.zeros(shape=self.aued_mwh.shape)

        self.cap_op = self.aued_mwh - self.individual_supplied  # non-supplied areas where both capital and operational expenditure are required

        # save_results_normal.write_tiff(self.individual_supplied, 'individual_supplied_xxx')
        save_results_normal.write_tiff(self.aued_mwh, self.gt, 'aued_mwh',self.output_directory)  # nactual useful energy demand

        return self.individual_supplied, self.cap_op

    def pipe_length_plot_ratio(self):

        # relationship between demand and GFA_total
        # (the hotmpas cooling demand to GFA_total translated to actual demand and cooled GFA
        # y = 24.99841731328322x + 21.581936372528162

        # self.cooled_gfa = np.zeros_like(self.aued_mwh)
        # non_zero_mask = np.where(self.aued_mwh != 0)
        # self.cooled_gfa[non_zero_mask] = 24.99841731328322 * self.aued_mwh[non_zero_mask] + 21.581936372528162
        # save_results_normal.write_tiff(self.cooled_gfa, 'cooled_gfa_xxxxx')

        plot_ratio = self.gfa_m2 / 10000

        percentile = np.percentile(plot_ratio[plot_ratio != 0], 50)

        effective_width_distribution = np.where(np.logical_and(plot_ratio > percentile, plot_ratio != 0),
                                                np.exp(4) / plot_ratio, np.where(plot_ratio == 0, 0, np.exp(4)))

        effective_width_service = np.where(np.logical_and(plot_ratio > percentile, plot_ratio != 0),
                                           np.exp(4) / plot_ratio, np.where(plot_ratio == 0, 0,
                                                                            (np.log(plot_ratio) + 3.5) / (np.exp(
                                                                                0.7737 + 0.18559 * np.log(
                                                                                    plot_ratio)))))

        effective_width_service = np.where(effective_width_service < 0, 0, effective_width_service)

        # distribution pipe length in meters
        pipe_length_distribution = np.where(effective_width_distribution == 0, 0, 10000 / effective_width_distribution)


        pipe_length_service = np.where(effective_width_service == 0, 0, 10000 / effective_width_service)


        # TODO : Also need to add service pipe after confirming numbers
        self.total_pipe_length_pr_m = pipe_length_distribution
        save_results_normal.write_tiff(self.total_pipe_length_pr_m, self.gt, 'total_pipe_length_pr_m',self.output_directory)

        return self.total_pipe_length_pr_m

    def pipe_length_estimation_method(self):
        if self.pipe_proxy == 1:
            self.pipe_length = self.total_pipe_length_pr_m
        elif self.pipe_proxy == 2:
            self.pipe_length = self.road_m

        return self.pipe_length

    def anchor_assumption(self):  # change_assumption
        peak_estimate_approach = 2  # 1 is CDD and 2 is FLH #(default:2 )
        #######################################################################################################################
        ##*** for all cells
        # estimate peak with TUED
        peak_demand_MW = anchor_1.peak_estimate(peak_estimate_approach, self.aued_mwh, self.flh_cooling_days)[0]
        save_results_normal.write_tiff(peak_demand_MW, self.gt, 'peak_demand_MW',self.output_directory)

        non_ind_areas_mwh = self.cap_op.copy()  # cap_op is already in terms of AUED

        cells = np.where(non_ind_areas_mwh != 0)

        self.linear_demand_density = np.zeros_like(non_ind_areas_mwh)
        self.linear_demand_density[cells] = non_ind_areas_mwh[cells] / self.pipe_length[
            cells]  # pipe length us based on total GFA
        save_results_normal.write_tiff(self.linear_demand_density, self.gt, 'linear_demand_density',
                                       self.output_directory)

        dia_meter_estimate = anchor_1.estimate_diameter(self.delta_T_dc, self.avg_fluid_velocity_mperS, peak_demand_MW,
                                                        self.linear_demand_density, scaling=True)
        diameter_mm = dia_meter_estimate[0]
        save_results_normal.write_tiff(diameter_mm, self.gt, 'diameter_mm',self.output_directory)

        volume_flow_rate_m3perS = dia_meter_estimate[1]
        save_results_normal.write_tiff(volume_flow_rate_m3perS, self.gt, 'volume_flow_rate_m3perS',self.output_directory)

        # grid investment are calculated for all cells # costs are not annualized
        grid_investment_costs_Eur = anchor_1.dc_cost_calculation(diameter_mm, self.pipe_length)[0]
        save_results_normal.write_tiff(grid_investment_costs_Eur, self.gt, 'grid_investment_costs_Eur',
                                       self.output_directory)
        grid_investment_unit_costs_Eurperm = anchor_1.dc_cost_calculation(diameter_mm, self.pipe_length)[1]

        # neighbour demand -->includes both potential anchors and neighbours
        # TODO: Neighbors could also be cells where demand is already supplied
        N_ued_mwh = (non_ind_areas_mwh / grid_investment_costs_Eur) * grid_investment_costs_Eur
        N_ued_mwh[np.isnan(N_ued_mwh)] = 0

        # peak only for the anchors and the neighbours
        N_peak_MW = (peak_demand_MW / N_ued_mwh) * N_ued_mwh

        #######################################################################################################################

        # 0.28 based on minimum avg value for existing DC girds in Vienna
        # N_ued_mwh = np.where(self.non_res_ratio < 0.28, 0, N_ued_mwh)
        # ave_results_normal.write_tiff(N_ued_mwh, 'xxx_yyy')

        potential_anchors_I_mwh = N_ued_mwh * (self.non_res_ratio > 0.2)

        ### **only anchors
        potential_anchors_I_mwh = potential_anchors_I_mwh * [
            potential_anchors_I_mwh > np.percentile(potential_anchors_I_mwh[potential_anchors_I_mwh > 0],
                                                    self.maximum_energy_threshold)]
        potential_anchors_I_mwh.shape = potential_anchors_I_mwh.shape[1:3]

        print('###########')
        # print(np.percentile(N_ued_mwh[N_ued_mwh > 0], self.maximum_energy_threshold))

        # readjusting the GFA default file to the cells of the potential_anchors_mwh
        gfa_m2_PA = (self.gfa_m2 / potential_anchors_I_mwh) * potential_anchors_I_mwh
        gfa_m2_PA[np.isnan(gfa_m2_PA)] = 0

        if self.anchor_definition_type == 1:
            gfa_m2_PA = gfa_m2_PA * [gfa_m2_PA > np.percentile(gfa_m2_PA[gfa_m2_PA > 0],
                                                               self.minimum_gfa_threshold)]  # '< for default assumption 1' change_assumption
        else:
            gfa_m2_PA = gfa_m2_PA * [gfa_m2_PA < np.percentile(gfa_m2_PA[gfa_m2_PA > 0], self.minimum_gfa_threshold)]

        gfa_m2_PA.shape = gfa_m2_PA.shape[1:3]
        # print(np.percentile(gfa_m2_PA[gfa_m2_PA > 0], self.minimum_gfa_threshold))


        # in MWh
        ## readjusting the potential_anchors_I_mwh to gfa_m2_PA
        ## PA_II is the file with the demand raster only for identified anchors
        PA_II = (potential_anchors_I_mwh / gfa_m2_PA) * gfa_m2_PA
        PA_II[np.isnan(PA_II)] = 0

        ## filtering potential anchors based on peak demand capacity
        PA_II_MW = (N_peak_MW / PA_II) * PA_II

        #######################################################################################################################
        # upper_threshold = np.percentile(PA_II_MW[~np.isnan(PA_II_MW)], self.anchor_MW_threshold[1])
        # lower_threshold = np.percentile(PA_II_MW[~np.isnan(PA_II_MW)], self.anchor_MW_threshold[0])
        # PA_II_MW = PA_II_MW * [(PA_II_MW > lower_threshold)] * [(PA_II_MW < upper_threshold)]
        #
        # # if self.anchor_definition_type == 1:
        # #     new_threshold = np.nanmax(PA_II_MW) * 0.7
        # #     PA_II_MW = PA_II_MW * [PA_II_MW > new_threshold]
        # #
        # # # if np.nanmax(PA_II_MW) < self.minimum_peak_threshold:
        # # #     new_threshold = np.nanmax(PA_II_MW) * 0.8
        # # #     PA_II_MW = PA_II_MW * [PA_II_MW > new_threshold]
        # # # else:
        # # #     PA_II_MW = PA_II_MW * [
        # # #         PA_II_MW > self.minimum_peak_threshold]  ## For vienna the minimum is 1.53 MW from appraoch 1
        # # # PA_II_MW = PA_II_MW * [PA_II_MW > 0.05] ## For vienna the minimum is 0.09 MW from appraoch 2
        # PA_II_MW.shape = PA_II_MW.shape[1:3]
        # PA_II_MW[np.isnan(PA_II_MW)] = 0
        # # ## PA_II re-filtered to get align with PA_II_MW
        # PA_II = (PA_II / PA_II_MW) * PA_II_MW
        # PA_II[np.isnan(PA_II)] = 0
        # # PA_II is the annual demand of all the potential anchors in MWh
        # anchor_point_count = np.count_nonzero(PA_II)
        #######################################################################################################################

        ## grid_investment_costs only for the anchors
        # A_ued_mwh would be the same as PA_II (at least for default case--recheck for other cases)
        A_aued_mwh = (N_ued_mwh / PA_II) * (PA_II)
        A_aued_mwh[np.isnan(A_aued_mwh)] = 0

        ## grid investment cropped down to only anchor points
        grid_investment_costs_anchors = grid_investment_costs_Eur / A_aued_mwh * A_aued_mwh
        grid_investment_costs_anchors[np.isnan(grid_investment_costs_anchors)] = 0
        grid_investment_unit_costs_Eurperm = grid_investment_unit_costs_Eurperm / A_aued_mwh * A_aued_mwh
        grid_investment_unit_costs_Eurperm[np.isnan(grid_investment_unit_costs_Eurperm)] = 0
        #######################################################################################################################
        # Grid linear density
        # linear_density_mwhperm = N_tued_mwh / road_m
        # linear_density_mwhperm[np.isnan(linear_density_mwhperm)] = 0
        # save_results.write_tiff(linear_density_mwhperm,'linear_density_mwhperm')

        ## if no optimization is done and grids are only anchors
        print('Total number of anchors: {}'.format(np.sum(A_aued_mwh > 0)))
        # print('Total Demand met: {} MWh'.format(PA_II.sum()))
        print('Total Demand of the identified anchors: {} MWh'.format(A_aued_mwh.sum()))
        # print('% of Demand met: {} %'.format(np.round((PA_II.sum() / non_ind_areas_mwh.sum()) * 100, 3)))
        # also includes areas where DC is not technologically feasible. example: cells without roads
        print('% of Demand assessed: {} % (Vs. all actual demand in the region)'.format(
            np.round((A_aued_mwh.sum() / non_ind_areas_mwh.sum()) * 100, 3)))
        # print('Total Investment: € {}'.format(np.round(grid_investment_costs_anchors.sum())))
        # in case when the expansion is zero
        print(
            'Total Investment for grid (max Possible-if only all anchors are converted to individual DC grids): Euro {}'.format(
                np.round(grid_investment_costs_anchors.sum())))
        print(
            '############################################################################################################')
        return (
            non_ind_areas_mwh, diameter_mm, grid_investment_costs_Eur, grid_investment_unit_costs_Eurperm, N_ued_mwh,
            N_peak_MW, PA_II, PA_II_MW)

    def set_anchor_assumption_values(self):
        individual_supplied, cap_op = self.working_area_identification()
        total_pipe_length_pr_m = self.pipe_length_plot_ratio()
        pipe_length = self.pipe_length_estimation_method()
        (non_ind_areas_mwh, diameter_mm, grid_investment_costs_Eur, grid_investment_unit_costs_Eurperm, N_ued_mwh,
         N_peak_MW, PA_II, PA_II_MW) = self.anchor_assumption()

        self.non_ind_areas_mwh = non_ind_areas_mwh
        self.diameter_mm = diameter_mm
        self.grid_investment_costs_Eur = grid_investment_costs_Eur
        self.grid_investment_unit_costs_Eurperm = grid_investment_unit_costs_Eurperm
        self.N_ued_mwh = N_ued_mwh
        self.N_peak_MW = N_peak_MW
        self.PA_II = PA_II
        self.PA_II_MW = PA_II_MW
        self.total_pipe_length_pr_m = total_pipe_length_pr_m
        self.pipe_length = pipe_length

    def individaul_LCOC_anchors(self):
        ###############################################################################################################
        # cells where cooling supply systems don't exist at all
        self.LCOC_ind_cap_op = individual_costs.individual_supply_calculations_inv_new(self.cap_op, self.af_ind,
                                                                                       self.cf_ind,
                                                                                       self.flh_cooling_days,
                                                                                       self.interest_rate,
                                                                                       self.depreciation_ac,
                                                                                       self.electricity_prices,
                                                                                       self.ind_tec_SEER)

        # if not os.path.exists(
        #         read_inputs.input_directory2 + 'LCOC_ind_all_' + save_results_normal.current_time[-8:-3] + '.tif'):
        #     save_results_normal.write_tiff(self.LCOC_ind_cap_op, 'LCOC_ind_all')
        # else:
        #     pass
        #
        # self.LCOC_ind_cap_op = np.nan_to_num(self.LCOC_ind_cap_op, 0)
        #
        # self.average_capop_LCOC = self.LCOC_ind_cap_op[
        #     self.LCOC_ind_cap_op != 0].mean()  # average LCOC individual system (AC)
        # where the demand is 100% not met
        self.average_capop_LCOC = self.LCOC_ind_cap_op

        return self.LCOC_ind_cap_op, self.average_capop_LCOC
