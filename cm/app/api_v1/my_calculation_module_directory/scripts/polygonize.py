import os
import numpy as np
import pandas as pd
import geopandas as gpd

from osgeo import gdal, ogr,osr
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
from scipy.ndimage import measurements

def symbol_list_creation(raster_for_symbology):
    """
    :param raster_for_symbology: the raster layer for which symbology is to be developed
    :return: based on the values in the raster layer categories the cells into 5 equal parts for visualization
    """

    min_val_dh, max_val_dh = np.min(raster_for_symbology), np.max(raster_for_symbology)

    diff = max_val_dh - min_val_dh
    # calculate the teps in legend. 6 steps are generated
    if diff > 0:
        if min_val_dh > diff / 4:
            symbol_vals = [round(min_val_dh - diff / 8 + i * diff / 4, 1) for i in range(5)]
        else:
            symbol_vals = [round(min_val_dh + i * diff / 4, 1) for i in range(5)]
    else:
        # synthesis_diff of 1 GWh
        synthesis_diff = 1
        symbol_vals = [round(min_val_dh - 1 + i * synthesis_diff / 2, 1) for i in range(5)]
    symbol_vals_str = [str(item) for item in symbol_vals]

    return symbol_vals_str


def polygonize(raster_layer_name,directory,output_name):
    '''

    :param raster_layer_name: array
    :param directory: directory to be saved in
    :param output_name: name of the output shape file
    :return:
    '''
    raster = gdal.Open(directory + raster_layer_name)
    band = raster.GetRasterBand(1)
    srs = osr.SpatialReference()

    #band = input_array
    epsg = 3035
    srs.ImportFromEPSG(epsg)
    shpDriver = ogr.GetDriverByName('ESRI Shapefile')

    #output_shp1 = directory + output_name
    output_shp1 = output_name

    if os.path.exists(output_shp1):
        shpDriver.DeleteDataSource(output_shp1)
    outDataSource = shpDriver.CreateDataSource(output_shp1)
    outLayer = outDataSource.CreateLayer('outPolygon', srs,
                                         geom_type=ogr.wkbPolygon)
    newField = ogr.FieldDefn('FID', ogr.OFTInteger)
    outLayer.CreateField(newField)
    # polygonize
    gdal.Polygonize(band, band, outLayer, 0, options=["8CONNECTED=8"])
    # save layer
    outDataSource = outLayer = band = None

    return None


#def polygonize_np()



def estimate_head_loss(average_diameter_in_mm,grid_length):
    # head_loss = f*l*(v**2)/2*d*g
    friction_coefficeint = 0.019 # f
    #average_length_m = 10000 # l # asuumed that source must be at least 10 km from the grid # thus the costs calculated are maximum
    average_length_m = grid_length * 2 # assuming pump to be sufficeint for supply and return pipes. Supply to point has not been considered.
    avg_fluid_velocity_mperS = 3 # v # based on literature and also used in scripts.anchor_1.estimate_diameter()
    g_mpers2 = 9.81 #g

    head_loss_m = (friction_coefficeint * average_length_m * avg_fluid_velocity_mperS**2)/(2 * (average_diameter_in_mm/1000) * g_mpers2)

    return head_loss_m

def estimate_pump_size(head_loss_m, volume_flow_rate_m3perS):
    #Pump_power_kw = (volume_flow_rate_m3perh * density_of_water_kgperm3 * g_mpers2 * head_loss_m) / (efficiency * 1000*3600)
    #https://www.engineeringtoolbox.com/pumps-power-d_505.html
    density_of_water_kgperm3 = 1000 # rho
    g_mpers2 = 9.81  # g
    pump_efficiency = 0.8

    volume_flow_rate_m3perh = volume_flow_rate_m3perS * 3600
    Pump_power_kw = (volume_flow_rate_m3perh * density_of_water_kgperm3 * g_mpers2 * head_loss_m) / (
                pump_efficiency * 1000 * 3600)

    return Pump_power_kw


def pump_cost_estimation(pump_size_kw, electricity_price):
    ## pump cost curve : Relations.xlsx Sheet_9
    #capex = 38365 * np.exp((0.006 * pump_size_kw))

    capex = 38365 * np.exp((0.006 * pump_size_kw))

    r = 0.06
    T = 20
    crf = (r * (1 + r) ** T) / (((1 + r) ** T) - 1)
    annualized_capex_EUR = capex * crf

    af = 0.9
    cf = 0.7
    flh = 60 #days;# same as anchor assumption
    opex_Eur = pump_size_kw * af * cf * flh * 24 * electricity_price

    annualized_pump_total_Eur = annualized_capex_EUR + opex_Eur

    return annualized_pump_total_Eur

# shape_file_name = output_polygon_name
# directory = output_directory
# electricity_price = electircity_price_EurpKWh

def cluster_parameters_allocation(changing_parameter,shape_file_name,run_start_time,directory,electricity_price):

    suffix = '_' + run_start_time[-8:-3] + '.tif'
    suffix_with_param = '_' + run_start_time[-8:] + '_' + str(changing_parameter) + '.tif'

    input_directory = directory

    # Load vector layer
    vector = gpd.read_file(shape_file_name)

    cluster_parameters = {'Avg_dia': 'diameter_mm' + suffix,
                          'Cell_count': 'cluster_default' + suffix_with_param,
                          'Avg_flow': 'volume_flow_rate_m3perS' + suffix,
                          'Tot_dem': 'Demand_met_by_expansion' + suffix_with_param,
                         'Avg_LCOCgr': 'levl_dist_grid_cost_per_mwh' + suffix,
                          'Avg_LCOCin': 'LCOC_ind_clusters' + suffix_with_param,
                          'Tot_inv_g': 'total_investment_grid' + suffix_with_param,
                          'GFA_m2': 'cooling_gfa' + suffix_with_param,
                          'grid_len': 'network_length' + suffix_with_param}

    aggregation_type = {'sum':['Cell_count','Tot_dem','Tot_inv_g','GFA_m2','grid_len'] ,
                       'mean':['Avg_dia', 'Avg_flow', 'Avg_LCOCgr','Avg_LCOCin']}

    new_cluster_parameters = ['head_loss', 'Pump_sizing', 'Pump_costs_CAPEX', 'Pump_costs_OPEX', 'Total_pump_costs',
                              'Total_grid_costs']

    for main_key in cluster_parameters.keys():
        raster_path = input_directory + cluster_parameters[main_key]
        masked_raster, masked_transform = mask(rasterio.open(raster_path), vector.geometry, crop=True)
        masked_raster.shape = masked_raster.shape[1:3]
        aggregation = [key for key, value in aggregation_type.items() if main_key in value][0]
        stats = zonal_stats(vector, masked_raster, affine=masked_transform, nodata=0, stats =aggregation)
        vector[main_key] = [stat[aggregation] for stat in stats]

    ## for weighted levelized cost
    raster_path_1 = directory + 'levl_dist_grid_cost_per_mwh' + suffix
    masked_raster_1, masked_transform_1 = mask(rasterio.open(raster_path_1), vector.geometry, crop=True)
    masked_raster_1.shape = masked_raster_1.shape[1:3]

    raster_path_2 = directory + 'aued_mwh' + suffix
    masked_raster_2, masked_transform_2 = mask(rasterio.open(raster_path_2), vector.geometry, crop=True)
    masked_raster_2.shape = masked_raster_2.shape[1:3]

    wt_avg_num = masked_raster_1 * masked_raster_2
    stats_1 = zonal_stats(vector, wt_avg_num, affine=masked_transform_1, nodata=0, stats='sum')
    stats_2 = zonal_stats(vector, masked_raster_2, affine=masked_transform_2, nodata=0, stats='sum')
    numerator = np.array([stats['sum'] for stats in stats_1])
    denominator = np.array([stats['sum'] for stats in stats_2])
    vector['Wt_LCOCgr'] =numerator/denominator




    df = pd.DataFrame()
    df.loc[:, 'head_loss_m'] = vector.apply(lambda row: estimate_head_loss(row['Avg_dia'], row['grid_len']), axis=1)
    df.loc[:, 'volume_flow_rate_m3perS'] = vector.loc[:, 'Avg_flow'].values
    df.loc[:, 'Pump_power_kw'] = df.apply(
        lambda row: estimate_pump_size(row['head_loss_m'], row['volume_flow_rate_m3perS']), axis=1)
    df.loc[:, 'Total_pump_cost'] = df.apply(lambda row: pump_cost_estimation(row['Pump_power_kw'],electricity_price), axis=1)

    FLH_cooling_days = 60
    af = 0.9
    cf = 0.5
    flh = FLH_cooling_days * 24 * af * cf

    vector.loc[:,'Peak_MW'] = vector.Tot_dem / flh

    vector.Peak_kW = vector.Peak_MW * 1000 #kJ/s
    Cp = 4.187 #KJ/kgK
    delta_T = 9 #°C
    mass_flow = vector.Peak_kW/(Cp * delta_T) # in kg/s
    volume_flow = mass_flow * 0.001 # in m3/s (cms)
    vector.loc[:,'m3Ps'] = volume_flow.values

    r = 0.06
    T = 30
    crf = (r * (1 + r) ** T) / (((1 + r) ** T) - 1)
    vector.loc[:, 'Inv_g_ann'] = vector.Tot_inv_g * crf
    vector.loc[:, 'PumpAnnEur'] = df.loc[:, 'Total_pump_cost'].values

    vector.loc[:,'Inv_gp'] = vector.Inv_g_ann + vector.PumpAnnEur #Total investemnt Euros annualized
    vector.loc[:,'LCOCgrid'] = vector.Inv_gp / vector.Tot_dem ##overall LCOC EUR/MWh
    vector.loc[:,'LDD'] = vector.Tot_dem / vector.grid_len ## average linear density per cluster
    vector.loc[:,'spc_dem'] = vector.Tot_dem / vector.GFA_m2 ## average specific demand per cluster

    vector.loc[:,'Area'] = vector.geometry.area
    #vector = vector[vector.Area > 30000]

    vector.drop(columns=['FID','Area'], inplace=True)

    vector.to_file(shape_file_name)
    return None


def label_clusters(raster_layer):
    struct = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])

    clusters, cluster_count = measurements.label(raster_layer, structure=struct)

    return clusters, cluster_count














#
# cluster_parameters = {'Average_diameter': 'diameter_mm' + suffix,
#                       'Cell_count': 'cluster_default' + suffix_with_param,
#                       'Average_vol_flow_rate': 'volume_flow_rate_m3perS' + suffix,
#                       'Total_demand': 'Demand_met_by_expansion' + suffix_with_param,
#                       'Average_grid_LCOC': 'LCOC_DC' + suffix_with_param,
#                       'Average_ind_LCOC': 'LCOC_ind_clusters' + suffix_with_param,
#                       'Total_investment_grid': 'total_investment_grid' + suffix_with_param,
#                       'GFA_m2': 'cooling_gfa' + suffix_with_param,
#                       'grid_length': 'network_length' + suffix_with_param}
#
# aggregation_type = {
#     'sum': ['Cell_count', 'Total_demand', 'Total_investment_grid', 'grid_length', 'GFA_m2', 'grid_length'],
#     'mean': ['Average_diameter', 'Average_vol_flow_rate', 'Average_grid_LCOC', 'Average_ind_LCOC']}

# directory = 'G:\\My Drive\\TU WIEN\\Work\\PhD\M1\\results_2\\2023-04-13-16-39-46\\'
#
# vector_file = gpd.read_file(directory + 'polygon_16-39-46_261.shp')
#
# vector_file.rename(columns= {'Average_di':'Avg_dia', 'Average_vo':'Avg_flow', 'Total_dema':'Tot_dem',
#        'Average_gr':'Avg_LCOCgr', 'Average_in':'Avg_LCOCin', 'Total_inve':'Tot_inv_g', 'grid_lengt':'grid_len'}, inplace =True)