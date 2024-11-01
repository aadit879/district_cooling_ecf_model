import numpy as np
from osgeo import gdal
import os
import datetime


file_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(file_directory, '..'))

# output_directory = root_directory + '\\Output\\'
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def write_tiff(input_array, gt_base, file_name, changing_parameter, output_directory,
               point=[0, 0], current_time=current_time):
    '''

    :param input_array: np.ndarray to be saved as tiff
    :param gt_base: geotransform of the working area raster
    :param file_name: name of the saved tiff file (without .tif)
    :param output_directory: directory for saving
    :param point: top_left_coordinates of the raster
    :return: None. The file is created in the directory
    '''
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    # Create a new directory with the current time and date

    directory = os.path.join(output_directory, current_time)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, file_name + '_' + current_time[-8:] + '_' + str(changing_parameter) + '.tif')
    outds = driver.Create(file_path, xsize=input_array.shape[1], ysize=input_array.shape[0], bands=1,
                          eType=gdal.GDT_Float64)

    # hotmaps default information
    # Vienna
    # gt = (4780100.0, 100.0, 0.0, 2821800.0, 0.0, -100.0)

    x = gt_base[0] + point[1] * 100
    y = gt_base[3] - point[0] * 100

    # x = 4780100.0 + point[1] * 100
    # y = 2821800.0 - point[0] * 100

    gt = (x, 100.0, 0.0, y, 0.0, -100.0)
    proj = 'PROJCS["ETRS89-extended / LAEA Europe",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",52],PARAMETER["longitude_of_center",10],PARAMETER["false_easting",4321000],PARAMETER["false_northing",3210000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Northing",NORTH],AXIS["Easting",EAST],AUTHORITY["EPSG","3035"]]'

    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(input_array)
    outband.SetNoDataValue(np.nan)
    outband.FlushCache()
    # close datasets and bands
    outband = None
    outds = None

    return None