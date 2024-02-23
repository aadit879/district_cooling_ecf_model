from osgeo import gdal
from . save_results_normal import current_time

import os

#input_directory = 'G:\\My Drive\\TU WIEN\\Work\\PhD\\M1\\Input Data\\'
file_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(file_directory, '..'))
input_directory = root_directory + '\\Input\\'
input_directory2 = root_directory + '\\Output\\' + current_time + '\\'

def read_raster(file_name, input_directory = input_directory):
    src_1 = gdal.Open(input_directory + file_name)
    src_1.RasterCount
    gt = src_1.GetGeoTransform()
    proj = src_1.GetProjection()
    band = src_1.GetRasterBand(1)
    array = band.ReadAsArray()
    return [gt,proj,band,array]