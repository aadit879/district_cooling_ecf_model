
import matplotlib.pyplot as plt


# plot to check the results
def array_plotter(array):
    plt.imshow(array)
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def raster_distribution_plot(raster,title,x_label, nonzero_values = True):
    # Plot a histogram of the 2D numpy array
    if nonzero_values == True:
        nonzero_values = raster[raster != 0]
        plt.hist(nonzero_values, bins=100)
    else:
        plt.hist(raster.flatten(), bins=100)

    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.title(title)

    # Display the histogram
    plt.show()

    return None

#raster_distribution_plot(electricity_consumption, 'electricity_consumption_for_cooling','MWh per annum')
#raster_distribution_plot(LCOC_ind, 'LCOC Individual','Eur/MWh')
#raster_distribution_plot(LCOC_ind_cap_op , 'LCOC Individual','Eur/MWh')
#raster_distribution_plot(LCOC_ind_op , 'LCOC Individual','Eur/MWh')
#raster_distribution_plot(LCOC_ind , 'LCOC Individual','Eur/MWh')