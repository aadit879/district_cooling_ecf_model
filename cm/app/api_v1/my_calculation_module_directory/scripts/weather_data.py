import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from scripts.save_results_normal import output_directory

import_directory = 'Z:\\Ongoing Stuff\\Data and stuff\\Weather Data\\Wien2016\\'


def copernicus_temp_extraction(year, import_directory):
    weather_files = os.listdir(import_directory)
    annual_average_temp = []
    for filename in weather_files:
        full_name = import_directory + filename
        data = Dataset(full_name, 'r')

        tas_data = data.variables['tas'][:]

        tas_data = tas_data - 273

        mean_temp = np.mean(np.mean(tas_data, axis=2), axis=1)
        annual_average_temp.append(mean_temp)

    annual_average_temp = np.array(annual_average_temp, dtype=object)
    annual_average_temp = ma.concatenate(
        [ma.masked_array(a, mask=a.mask, fill_value=a.fill_value, dtype=a.dtype) for a in annual_average_temp])

    total_length = len(annual_average_temp)
    is_leap_year = total_length / 24 == 366

    if is_leap_year:
        start_date = pd.Timestamp(year=year, month=1, day=1)
        time_index = pd.date_range(start=start_date, periods=total_length, freq='H')
    else:
        start_date = pd.Timestamp(year=year, month=1, day=1)
        time_index = pd.date_range(start=start_date, periods=total_length, freq='H')

    temp_profile = pd.DataFrame({'Avg_temp_C': annual_average_temp}, index=time_index)
    temp_profile.to_csv(output_directory + 'Vienna_temp_profile.csv')

    temp_profile.plot()
    plt.show()

    return None


def supply_water_temperature(output_directory):
    air_temp = pd.read_csv(output_directory + 'Vienna_temp_profile.csv', index_col=0, parse_dates=True)
    air_temp = air_temp.resample('D').mean()

    avg_daily_river_temp = 5 + 0.75 * air_temp.Avg_temp_C.values
    # Source: Relationship between Water Temperatures and Air Temperatures fro Central US Streams Eric B Preud'homme 1992

    river_temp_daily = pd.DataFrame({'River_temp_C': avg_daily_river_temp}, index=air_temp.index)

    plt.plot(air_temp.index, air_temp['Avg_temp_C'], label='Air Temperature')
    plt.plot(river_temp_daily.index, river_temp_daily['River_temp_C'], label='River Temperature')
    plt.legend()
    plt.show()

    river_temp_daily.to_csv(output_directory +  'Vienna_river_temp_daily.csv')
    return None

if __name__ == "__main__":
    copernicus_temp_extraction(2016, import_directory)
    supply_water_temperature(output_directory)
