# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


# pre_dev_datafile_folder = "Per_dev_csv/LR/"
# pre_dev_datafile_folder = "Per_dev_csv/Case1_L1_N1/"
pre_dev_datafile_folder = "Per_dev_csv/Case2_L1_N1/"
pre_dev_datafile_names = [
# "initLR-0.01_lrUC-5_lrUP-5000_L1-3_N1-173_L2-1_N2-3_2019_7_6_22_41_45_1"
"initLR-0.01_lrUC-5_lrUP-5000_L1-8_N1-175_L2-1_N2-3_2019_7_7_0_38_52_1"
                    ]
pre_dev_datafile_ext = ".csv"

dimInput = 2
dimOutput = 6
numN = 50
numE = 100
numDataPts = numN*numE     # total number of data points

############################################### Loading Data ###############################################

# Function for loading data
def load_data(fl_nm):
    ''' Returns numpy array of shape (total data points, dimInput+dimOutput) '''
    data_file = open(fl_nm, 'r')
    data_list = []
    for row in data_file.readlines():
        row_list = row.split(',')
        # print(row_list)
        for i in range(len(row_list)):
            try:
                row_list[i] = float(row_list[i])
            except:
                row_list.pop()
        data_list.append(row_list)
    data_file.close()   
    return np.array(data_list, dtype = np.float32, ndmin = 2)

############################################################################################################



for j in range(len(pre_dev_datafile_names)):
    data_arr_pre_dev  = load_data(pre_dev_datafile_folder+pre_dev_datafile_names[j]+pre_dev_datafile_ext)
    assert data_arr_pre_dev.shape == (numDataPts, dimInput+dimOutput), data_arr_pre_dev.shape

    data_input_pre_dev, data_output_pre_dev = np.split(data_arr_pre_dev, [dimInput], axis=1)

    # Making meshgrid.
    n = data_input_pre_dev[::numE, 0].reshape(numN)
    # print(n)
    e = data_input_pre_dev[-numE:, 1].reshape(numE)
    # print(e)
    Xn, Ye = np.meshgrid(n, e, indexing="ij")

    fig = plt.figure(figsize=plt.figaspect(1.5))
    fig.suptitle(pre_dev_datafile_names[j])


    Z = np.empty((dimOutput, numN, numE))

    for i in range(numN):
    	for j in range(numE):
    		Z[:, i, j] = (data_output_pre_dev[numE*i+j, :])

    # Plot the surface.
    clrs = [cm.seismic, cm.seismic, cm.seismic, cm.RdGy, cm.RdGy, cm.RdGy]
    for i in range(6):
        ax = fig.add_subplot(3, 2, i+1, projection='3d')
        surf = ax.plot_surface(Xn, Ye, Z[i], cmap=clrs[i], linewidth=1, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(0, 100)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # # Add a color bar which maps values to colors.
    # # fig.colorbar(surf, shrink=0.5, aspect=5)

    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    plt.show()
