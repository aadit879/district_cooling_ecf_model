import numpy as np
import pandas as pd

from scripts import read_inputs

import warnings

#warnings.filterwarnings('ignore')

##delete
from scripts import save_results_normal as save_results
from scripts.pipe_cost_data import data as pipe_cost_data

def peak_estimate(approach, tued_mwh, FLH_cooling_days , T_ambient=21, T_set=15, af=0.9, cf=0.5):
    '''
    Source : http://people.tamu.edu/~i-choudhury/335_19.html
    :param approach: 1 = CDD appraoch and 2 = FLH approach
    :param T_ambient: # average dry bulb temperature for vienna https://weatherspark.com/y/81358/Average-Weather-in-Vienna-Austria-Year-Round
    :param T_set: # degree_centigrade # set at 18°C but much lower for commercial buildings
    :param af: technology availability factor (high value indicates reliability)
    :param cf: technology capacity factor (indicates the utilization rate of the technology; higher value indicates
                                            the efficient utilization of technology ) default 0.5
    :return: raster with peak demand per cell
    '''
    if approach == 1:
        # delta_t = T_ambient - T_set
        # peak_demand_MW = tued_mwh / (24 * (cdd_kdays / delta_t))
        # peak_demand_MW[np.isnan(peak_demand_MW)] = 0
        # FLH = 0
        print('CDD is to be provided as input. Check model_ecf version')
    else:

        FLH = FLH_cooling_days * 24 * af * cf
        peak_demand_MW = tued_mwh / (FLH)

    return peak_demand_MW, FLH


# *** for both anchors and neighbours
##the volume flow rate for each cell
def estimate_diameter(delta_T_dc, avg_fluid_velocity_mperS, peak_demand_MW, linear_demand_density, scaling):
    # gives diameter values in standard DN sizes

    peak_demand_TR = peak_demand_MW * 1000 * 0.284
    conversion_factor = 0.0000630902  # 1 gpm = 0.0000630902 m3/s
    #for all cells
    volume_flow_rate_m3perS = ((24 * peak_demand_TR) / (1.8 * delta_T_dc)) * conversion_factor
    # diameter_m: The average diameter of the pipe needed to supply the demand of the cell
    ## calculated for all cells
    diameter_m = np.sqrt((4 * volume_flow_rate_m3perS) / (avg_fluid_velocity_mperS * 3.14))
    diameter_mm_physical = diameter_m * 1000  ## for all cells

    pipe_costs = pd.DataFrame(pipe_cost_data)


    if scaling == True:
        cells = np.where(linear_demand_density > 0)
        factor = np.zeros_like(linear_demand_density)
        factor[cells] = - 0.3532 * np.log(linear_demand_density[cells]) + 2.6832

        diameter_mm = np.zeros_like(linear_demand_density)

        diameter_mm[cells] = diameter_mm_physical[cells] * factor[cells]


    i = pipe_costs.Size_DN.values[0] - 5
    diameter_mm = np.where(diameter_mm < i, 0, diameter_mm)  ## cells with diameter < 15 are filtered out
    # diameter_mm is is changed to a standard DN sizing
    for DN_size in pipe_costs.Size_DN.values:
        DN_values = list(pipe_costs.Size_DN.values)
        if DN_values.index(DN_size) == len(DN_values) - 1:
            pass
        else:
            current_size_index = DN_values.index(DN_size)
            next_size = DN_values[current_size_index + 1]
            j = (DN_size + next_size) / 2
            diameter_mm = np.where((diameter_mm >= i) & (diameter_mm < j), DN_size, diameter_mm)
            i = j

    return diameter_mm,volume_flow_rate_m3perS


# diameter_mm = estimate_diameter(peak_demand_MW)
#
# non_ind_areas_mwh = cap_op.copy()

def dc_cost_calculation(diameter_mm, pipe_length):
    ## also filters out areas where no roads are available
    # 1. grid investment costs are calculated for all potential neighbours
    pipe_costs = pd.DataFrame(pipe_cost_data)

    calculation_array = diameter_mm
    for DN_size in pipe_costs.Size_DN.values:
        grid_investment_costs = np.where((diameter_mm == DN_size),
                                         pipe_length * pipe_costs[pipe_costs.Size_DN == DN_size].iloc[0, 1],
                                         calculation_array)
        calculation_array = grid_investment_costs
    grid_investment_costs_Eur = calculation_array.copy()  # for all cells
    # save_results.write_tiff(grid_investment_costs_Eur,'grid_investment_costs_Eur')

    # 2. The second option is to calculate the unit length grid investment cost (not using road)
    calculation_array_2 = diameter_mm
    for DN_size in pipe_costs.Size_DN.values:
        grid_investment_unit_costs = np.where((diameter_mm == DN_size),
                                              pipe_costs[pipe_costs.Size_DN == DN_size].iloc[0, 1], calculation_array_2)
        calculation_array_2 = grid_investment_unit_costs
    grid_investment_unit_costs_Eurperm = calculation_array_2  # for all cells

    return [grid_investment_costs_Eur, grid_investment_unit_costs_Eurperm]


def Levl_dist_grid(r,T, grid_investment_costs_Eur, non_ind_areas_mwh):
    '''
    This module includes only the costs associated to the distribution grid
    :param grid_investment_costs_Eur: cells with potential DC supply whose
                                    total investment costs on grid are calculated
    :param non_ind_areas_mwh: areas with currently non-existent individual cooling
                                supply
    :return: levelized costs of cooling for DC
    '''
    crf = (r * (1 + r) ** T) / (((1 + r) ** T) - 1)
    annualized_grid_investment_EurperA = grid_investment_costs_Eur * crf
    grid_expansion_costs_EurperMWh = annualized_grid_investment_EurperA / (non_ind_areas_mwh )
    grid_expansion_costs_EurperMWh[np.isnan(grid_expansion_costs_EurperMWh)] = 0
    grid_expansion_costs_EurperMWh = np.where(np.isinf(grid_expansion_costs_EurperMWh), 0,
                                              grid_expansion_costs_EurperMWh)  # nonzero_values = grid_expansion_costs_EurperMWh[grid_expansion_costs_EurperMWh!=0]  # save_results.write_tiff(grid_expansion_costs_EurperMWh,'grid_expansion_costs_EurperMWh')

    return grid_expansion_costs_EurperMWh

#def diameter_reestimation_cluster(shape_files,plot_ratio,anchor):
    # identify_anchor
    # cluster raster is already available

    #vector = gpd.read_file(output_directory + output_polygon_name)


    # calculate_new diameter for each non anchor cells
    # calculate average grid distribution costs for each cell
    # calculate the average of each cell


#
# grid_expansion_costs_EurperMWh = LCOC_DC(grid_investment_costs_Eur, non_ind_areas_mwh)

def zero_to_nan(arr):
    return np.where(arr == 0, np.nan, arr)
#
# # ## final outputs
# potential_demand_MWh = N_tued_mwh.copy()
# potential_demand_MWh = zero_nan(potential_demand_MWh)
# potenital_demand_MW = N_peak_MW.copy()
# # potenital_demand_MW = zero_nan(potenital_demand_MW)
#
# potential_anchors_MWh = PA_II.copy()
# potential_anchors_MWh = zero_nan(potential_anchors_MWh)
# potential_anchors_MW = PA_II_MW.copy()
# #potential_anchors_MW = zero_nan(potential_anchors_MW)
#
# grid_investment_unit_costs_Eurperm = grid_investment_unit_costs_Eurperm/grid_expansion_costs_EurperMWh * grid_expansion_costs_EurperMWh
# grid_expansion_costs_EurperMWh[np.isnan(grid_expansion_costs_EurperMWh)] = 0
# grid_expansion_costs_EurperMWh = zero_nan(grid_expansion_costs_EurperMWh)
#
#
# save_results.write_tiff(potential_demand_MWh,'potential_demand_MWh')
# save_results.write_tiff(potenital_demand_MW,'potenital_demand_MW')
# save_results.write_tiff(potential_anchors_MWh,'potential_anchors_MWh')
# save_results.write_tiff(potential_anchors_MW,'potential_anchors_MW')
# save_results.write_tiff(grid_expansion_costs_EurperMWh,'grid_expansion_costs_EurperMWh_5')
# save_results.write_tiff(grid_investment_unit_costs_Eurperm,'grid_investment_unit_costs_Eurperm')
#
