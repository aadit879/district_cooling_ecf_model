import numpy as np

import random

from . read_inputs import read_raster

import warnings

warnings.filterwarnings('ignore')

def random_selector(tued_mwh, gfa_m2, per_actual_demand):
    '''
    The function selects cells where individual cooling supply currently exists. Highest demand areas with large non-residential GFA.
    :param tued_mwh: the overall theoretical useful energy demand of the region
    :param gfa_m2: the overall gross floor area of non-residential buildings
    :param per_actual_demand: ratio of actual covered demand to theoretical demand
    :return: raster layer where individual systems supply cooling
    '''

    global covered_demand_raster
    actual_demand = round(tued_mwh.sum() * per_actual_demand, 0)

    # Initialize variables
    covered_demand = 0
    i = 0

    # Create a mask to keep track of removed cells
    removed_mask = np.zeros(tued_mwh.shape, dtype=bool)

    while covered_demand < actual_demand:

        percentile_tued = np.percentile(tued_mwh[tued_mwh > 0], 99 - i)
        percentile_gfa = np.percentile(gfa_m2[gfa_m2 > 0], 99 - i)

        mask_tued = tued_mwh >= percentile_tued
        mask_gfa = gfa_m2 >= percentile_gfa

        combined_mask = np.logical_and(mask_tued, mask_gfa)

        row, col = np.where(combined_mask)

        covered_demand_raster = np.zeros(tued_mwh.shape)
        covered_demand_raster[row, col] = tued_mwh[row, col]

        covered_demand = covered_demand_raster.sum()

        i += 1

        # Calculate the excess demand (positive or negative)
        excess_demand = covered_demand_raster.sum() - actual_demand

        if excess_demand > 0:
            # Find the indices of non-zero cells in ascending order of values
            non_zero_indices = np.where(covered_demand_raster > 0)
            sorted_indices = np.argsort(covered_demand_raster[non_zero_indices])
            sorted_row = non_zero_indices[0][sorted_indices]
            sorted_col = non_zero_indices[1][sorted_indices]

            # Create a mask for cells to be removed
            remove_mask = np.zeros_like(covered_demand_raster, dtype=bool)

            # Identify cells to be removed based on excess_demand
            cumulative_sum = np.cumsum(covered_demand_raster[sorted_row, sorted_col])
            removed_cells = cumulative_sum <= excess_demand

            remove_mask[sorted_row[removed_cells], sorted_col[removed_cells]] = True

            # Set values to zero for cells to be removed
            covered_demand_raster[remove_mask] = 0

    return covered_demand_raster


def individual_supply_calculations_inv_new(raster,af,cf,FLH_cooling_days,r,T, electircity_price_EurpKWh, SEER):
    '''
    :param raster:
    :param electircity_price_EurpKWh:
    :param SEER:
    :return: LCOC_ind: the LCOC for all cells where there are no existing cooling supply systems.
            The 30% of the cells with the exisiting supply are not accounted for

    '''
    ##assessment for Air Conditioning
    # technology availability (high value indicates reliability)
    flh = (FLH_cooling_days * 24 * af * cf)
    crf = (r * (1 + r) ** T) / (((1 + r) ** T) - 1)

    CAPEX = 286.12 * 1000   # EUR/MW (from Eff heating cooling pathways AC)
    OPEX_fix = 11954.7     # EUR/MW (from Eff heating cooling pathways AC)

    # the simplified equations avoids calculation of individual cells; that results the same.
    # Calculations available in the relations.xlxs, sheet simlify_ind
    LCOC_ind = CAPEX/flh * crf + OPEX_fix/flh + electircity_price_EurpKWh * 1000 / SEER

    return LCOC_ind


# cap_op = tued_mwh - individual_supplied
# LCOC_ind_cap_op = individual_supply_calculations_inv(cap_op)

# def individual_supply_calculations_inv_ext(raster, electircity_price_EurpKWh, SEER):
#     #TODO: Needs to be completely changed (refer individual_supply_calculations_inv_new)
#     '''
#     Calculation for those randomly identified cells where the units
#     are assumed to be pre-installed. No investment costs accounted.
#     :param raster: raster cells with the pre-identified individual supply
#     :return:
#     '''
#     ##assessment for Air Conditioning
#     # technology availability (high value indicates reliability)
#     af = 0.9
#     # technology capacity factor (higher value indicates efficeicny)
#     cf = 0.7
#     flh = 60
#     # TODO: Can the 8760 be further reduced to decrease the FLH? Check literature # where is the 24
#     maximum_capacity_MW = raster / (8760 * af * cf)
#
#     # investment_cost_EurpMW =  (300.58 * np.exp(-1.003 * maximum_capacity_MW)) * 1000
#     # investment_cost_EurpMW[np.where(investment_cost_EurpMW == investment_cost_EurpMW.max())]=0
#     # total_investment_costs_Eur = investment_cost_EurpMW * maximum_capacity_MW
#     r = 0.06
#     T = 20
#     crf = (r * (1 + r) ** T) / (((1 + r) ** T) - 1)
#     # investment_anualized_EUR = total_investment_costs_Eur * crf
#
#     # operation_cost_EurpMW = (12023 * np.exp(-1.003 * maximum_capacity_MW))
#     # operation_cost_EurpMW[np.where(operation_cost_EurpMW  == operation_cost_EurpMW .max())]=0
#     # annual_operation_cost_Eur = operation_cost_EurpMW * maximum_capacity_MW
#
#     # SEER = 3.6
#     electricity_consumption_mwh = raster / SEER
#     # electircity_price_EurpKWh = 0.44
#     annual_electricity_expense_Eur = electricity_consumption_mwh * electircity_price_EurpKWh * 100
#
#     total_annual_costs = annual_operation_cost_Eur + annual_electricity_expense_Eur
#
#     LCOC_ind = total_annual_costs / (raster * crf)
#     return LCOC_ind


def random_select_30pct(arr, anchor):
    arr_sum = arr.sum()
    target_sum = arr_sum * 0.3
    selected_sum = 0
    selected_indices = []
    indices = np.nonzero(anchor)
    indices = np.column_stack((indices[0], indices[1]))

    while selected_sum < target_sum:
        selected_index = random.choice(indices)
        selected_sum += arr[selected_index[0], selected_index[1]]
        selected_indices.append(selected_index)
        indices = np.delete(indices, np.argwhere(np.all(indices == selected_index, axis=1)), axis=0)

    selected_array = np.zeros_like(anchor)
    selected_array[np.array(selected_indices)[:, 0], np.array(selected_indices)[:, 1]] = 1

    return selected_array
