import os

from osgeo import gdal

from ..helper import generate_output_file_tif,generate_output_file_shp, create_zip_shapefiles
from ..constant import CM_NAME

import pandas as pd
import time

from . my_calculation_module_directory.Main import DC_identification

""" Entry point of the calculation module function"""

#TODO: CM provider must "change this code" # AM:DONE
#TODO: CM provider must "not change input_raster_selection,output_raster  1 raster input => 1 raster output"
#TODO: CM provider can "add all the parameters he needs to run his CM # AM:DONE
#TODO: CM provider can "return as many indicators as he wants" # AM:DONE

def calculation(output_directory, inputs_raster_selection,inputs_parameter_selection):
    #TODO the folowing code must be changed by the code of the calculation module
    '''
       # AM: all input parameters defined in the constant.py file are to be called here
       '''

    # input parameter
    # per_actual_demand = float(inputs_parameter_selection['per_actual_demand'])
    electricity_prices = float(inputs_parameter_selection['electricity_prices']) / 1000
    # maximum_energy_threshold = int(inputs_parameter_selection['maximum_energy_threshold'])
    # minimum_gfa_threshold =  int(inputs_parameter_selection['minimum_gfa_threshold'])
    # minimum_peak_threshold = int(inputs_parameter_selection['minimum_peak_threshold'])
    # anchor_MW_threshold =  int(inputs_parameter_selection['anchor_MW_threshold']) # TODO: AM: this needs to be adjusted to float from list
    flh_cooling_days = int(inputs_parameter_selection['flh_cooling_days'])
    COP_DC = float(inputs_parameter_selection['COP_DC'])
    delta_T_dc = int(inputs_parameter_selection['delta_T_dc'])
    # avg_fluid_velocity_mperS = int(inputs_parameter_selection['avg_fluid_velocity_mperS'])
    if 'ind_tec_SEER' in inputs_parameter_selection:
        ind_tec_SEER = float(inputs_parameter_selection['ind_tec_SEER'])
    else:
        print("'ind_tec_SEER' not found in inputs_parameter_selection")
    # af_ind = int(inputs_parameter_selection['af_ind'])
    # cf_ind = int(inputs_parameter_selection['cf_ind'])
    interest_rate = float(inputs_parameter_selection['interest_rate']) / 100
    if 'depreciation_dc' in inputs_parameter_selection:
        depreciation_dc = int(inputs_parameter_selection['depreciation_dc'])
    else:
        print("'depreciation_dc' not found in inputs_parameter_selection")

    depreciation_ac = int(inputs_parameter_selection['depreciation_ac'])

    # input raster layer
    in_raster_gfa_tot = inputs_raster_selection["gross_floor_area"]
    in_raster_gfa_non_res = inputs_raster_selection[
        "gross_floor_area"]  # TODO: AM: is this also correct for nonresidential
    in_raster_cdm = inputs_raster_selection["heat"]  # TODO : AM: check the type; should match constant.py

    # generate the output raster file

    output_raster_demand_covered = generate_output_file_tif(output_directory)
    output_raster_levl_grid_cost = generate_output_file_tif(output_directory)
    output_raster_network_length = generate_output_file_tif(output_directory)
    output_raster_grid_investment_cost = generate_output_file_tif(output_directory)
    output_raster_average_diameter = generate_output_file_tif(output_directory)

    output_shp = generate_output_file_shp(output_directory)

    theoretical_demand, actual_demand, \
    dc_coverage, graphics, summary_df, \
    cluster_shape, symbol_vals_str = DC_identification(electricity_prices, flh_cooling_days,
                                                       COP_DC, delta_T_dc,
                                                       ind_tec_SEER, interest_rate, depreciation_dc,
                                                       depreciation_ac, in_raster_gfa_tot,
                                                       in_raster_gfa_non_res, in_raster_cdm,
                                                       output_raster_demand_covered,
                                                       output_raster_levl_grid_cost, output_raster_network_length,
                                                       output_raster_grid_investment_cost,
                                                       output_raster_average_diameter, output_shp,
                                                       output_directory)

    ## AM: Outputs of the main.py

    # TODO to create zip from shapefile use create_zip_shapefiles from the helper before sending result
    # TODO exemple  output_shpapefile_zipped = create_zip_shapefiles(output_directory, output_shpapefile)

    result = dict()

    # total demand covered
    # percentage of the demand covered
    # raster layer showing the clusters

    result['name'] = CM_NAME
    result['indicator'] = [{"unit": "GWh", "name": "Total theoretical cooling demand in GWh within the selected zone",
                            "value": theoretical_demand},
                           {"unit": "GWh", "name": "Estimated actual cooling demand in GWh within the selected zone",
                            "value": actual_demand},
                           {"unit": "GWh", "name": "DC cooling potential in GWh within the selected zone",
                            "value": dc_coverage},
                           {"unit": "%",
                            "name": "Potential share of district cooling from total actual demand in selected zone",
                            "value": 100 * round(dc_coverage / theoretical_demand, 4)}
                           ]
    result['graphics'] = graphics

    if dc_coverage > 0:
        output_shp = create_zip_shapefiles(output_directory, output_shp)
        step = float(symbol_vals_str[4]) - float(symbol_vals_str[3])
        result["raster_layers"] = [
            {"name": "District Cooling areas - raster", "path": output_raster_demand_covered, "type": "custom",
             "symbology": [{"red": 254, "green": 237, "blue": 222, "opacity": 0.5, "value": symbol_vals_str[0],
                            "label": symbol_vals_str[0] + " GWh"},
                           {"red": 253, "green": 208, "blue": 162, "opacity": 0.5, "value": symbol_vals_str[1],
                            "label": symbol_vals_str[1] + " GWh"},
                           {"red": 253, "green": 174, "blue": 107, "opacity": 0.5, "value": symbol_vals_str[2],
                            "label": symbol_vals_str[2] + " GWh"},
                           {"red": 253, "green": 141, "blue": 60, "opacity": 0.5, "value": symbol_vals_str[3],
                            "label": symbol_vals_str[3] + " GWh"},
                           {"red": 230, "green": 85, "blue": 13, "opacity": 0.5, "value": symbol_vals_str[4],
                            "label": symbol_vals_str[4] + " GWh"},
                           {"red": 166, "green": 54, "blue": 3, "opacity": 0.5,
                            "value": str(float(symbol_vals_str[4]) + step),
                            "label": ">" + symbol_vals_str[4] + " GWh"}]
             },
            {"name": "Heat density map in potential DH areas - raster", "path": output_raster_levl_grid_cost,
             "type": "heat"
             },
            {"name": "Heat density map in potential DH areas - raster",
             "path": output_raster_network_length,
             "type": "heat"
             },
            {"name": "Heat density map in potential DH areas - raster",
             "path": output_raster_grid_investment_cost,
             "type": "heat"
             },
            {"name": "Heat density map in potential DH areas - raster",
             "path": output_raster_average_diameter,
             "type": "heat"
             }
            ]

        result['vector_layers'] = [
            {"name": "District Cooling areas and there potentials - shapefile", "path": output_shp, "type": "custom",
             "symbology": [{"red": 254, "green": 237, "blue": 222, "opacity": 0.5, "value": symbol_vals_str[0],
                            "label": symbol_vals_str[0] + " GWh"},
                           {"red": 253, "green": 208, "blue": 162, "opacity": 0.5, "value": symbol_vals_str[1],
                            "label": symbol_vals_str[1] + " GWh"},
                           {"red": 253, "green": 174, "blue": 107, "opacity": 0.5, "value": symbol_vals_str[2],
                            "label": symbol_vals_str[2] + " GWh"},
                           {"red": 253, "green": 141, "blue": 60, "opacity": 0.5, "value": symbol_vals_str[3],
                            "label": symbol_vals_str[3] + " GWh"},
                           {"red": 230, "green": 85, "blue": 13, "opacity": 0.5, "value": symbol_vals_str[4],
                            "label": symbol_vals_str[4] + " GWh"},
                           {"red": 166, "green": 54, "blue": 3, "opacity": 0.5,
                            "value": str(float(symbol_vals_str[4]) + step), "label": ">" + symbol_vals_str[4] + " GWh"}]
             }]

    print('result', result)
    return result


def colorizeMyOutputRaster(out_ds):
    ct = gdal.ColorTable()
    ct.SetColorEntry(0, (0,0,0,255))
    ct.SetColorEntry(1, (110,220,110,255))
    out_ds.SetColorTable(ct)
    return out_ds
