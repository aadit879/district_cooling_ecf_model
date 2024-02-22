import time

from . initialize import Parameters
from . Calculation_main import MainCalculation
import os

from rasterio.plot import show



def DC_identification(electricity_prices, flh_cooling_days, COP_DC, delta_T_dc, ind_tec_SEER, interest_rate,
                      depreciation_dc, depreciation_ac, in_raster_gfa_tot, in_raster_gfa_non_res, in_raster_cdm,
                      output_raster_demand_covered,
                      output_raster_levl_grid_cost, output_raster_network_length,
                      output_raster_grid_investment_cost,
                      output_raster_average_diameter, output_shp,
                      output_directory):

    start_time = time.time()

    params = Parameters(electricity_prices, flh_cooling_days, COP_DC, delta_T_dc, ind_tec_SEER, interest_rate,
                        depreciation_dc, depreciation_ac, in_raster_gfa_tot, in_raster_gfa_non_res, in_raster_cdm,output_directory)

    params.set_anchor_assumption_values()

    CM = MainCalculation(params)

    # for electircity_price_EurpKWh in params.electricity_prices:
    #     CM.all_runs(electircity_price_EurpKWh)

    clusters, anchor_df, Average_levl_dist_grid_cost_per_mwh,\
    network_length, anchor_to_cluster, LCOC_threshold, tot_demand_expan,\
    avg_LCOC_ind_clusters, tot_inv_grid, tot_gfa,\
    summary_df, cluster_shape, symbol_vals_str_1 = CM.all_runs(output_raster_demand_covered,output_raster_levl_grid_cost, output_raster_network_length,
                 output_raster_grid_investment_cost,  output_raster_average_diameter, output_shp)

    grid_investment_unit_costs_Eurperm = CM.grid_investment_unit_costs_Eurperm,
    grid_investment_costs_Eur = CM.grid_investment_costs_Eur

    theoretical_demand =summary_df.loc[0,'Theoretical Cooling Demand (TWh)']
    actual_demand = summary_df.loc[0,'Actual Cooling Demand (TWh)']
    dc_coverage = summary_df.loc[0,'DC Covered Demand (TWh)']

    graphics = [
        # {'type': 'bar',
        #  'xlabel': 'DC Area label',
        #  'ylabel': 'Potential (GWh/year)',
        #  "data": {"labels": [str(x) for x in range(1, 1+len(DHPot))],
        #                    "datasets": [{
        #                             "label": "Potential in coherent areas",
        #                             "backgroundColor": ["#3e95cd"]*len(DHPot),
        #                             "data": list(np.around(DHPot,2))
        #                             }]
        #             }
        #         },

        {'type': 'bar',
         'xlabel': "",
         'ylabel': 'Potential (GWh/year)',
         "data": {"labels": ['Annual Theoretical Cooling Demand', 'Annual Actual Cooling Demand','DC coverage potential'],
                           "datasets": [{
                                    "label": "DC Potential Assessment",
                                    "backgroundColor": ["#fe7c60", "#3e95cd",'#00FF00'],
                                    "data": [theoretical_demand, actual_demand, dc_coverage]
                                    }]
                    }
                }
    ]



    print("--- %s seconds ---" % (time.time() - start_time))



    return (theoretical_demand, actual_demand, dc_coverage, graphics, summary_df, cluster_shape, symbol_vals_str_1)

    # clusters, anchor_df, Average_levl_dist_grid_cost_per_mwh, network_length, anchor_to_cluster, LCOC_threshold,
    # tot_demand_expan, avg_LCOC_ind_clusters, tot_inv_grid, tot_gfa, grid_investment_unit_costs_Eurperm,
    # grid_investment_costs_Eur
# expected output ---> multiple cluter of cells (output 1)

## supply side optimization
# modelling the river
# optimze on possibility of connecting the clusters by the supply


# change the currentworking directory
# import os



