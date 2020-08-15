import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib

datafile_folder = "csv/"
datafile_name_act = "plot_data_d0-1_2_3_4_5_act"
datafile_name_pred = "plot_data_d0-1_2_3_4_5_pred"
datafile_ext = ".csv"

dimInput = 2       # input dimension
dimOutput = 6      # output dimension

numE_act = 5
numN_act = 15000
numDataPts_act = numE_act*numN_act     # total number of data points
numE_pred = 5
numN_pred = 11
numDataPts_pred = numE_pred*numN_pred     # total number of data points

plt_folder = "plots/"
plt_name = "plot_data_d0-1_2_3_4_5_pred"
num_subplot_rows = 3    # number of rows of subplot array in plot
num_subplot_cols = 2    # number of columns of subplot array in plot    

############################################### Loading Data ###############################################

# Function for loading data
def load_data(fl_nm):
    ''' Returns numpy array of shape (total data points, dimInput+dimOutput) '''
    data_file = open(fl_nm, 'r')
    data_list = []
    for row in data_file.readlines():
        row_list = row.split(',')
        for i in range(len(row_list)):
            row_list[i] = float(row_list[i])
        data_list.append(row_list)
    data_file.close()   
    return np.array(data_list, dtype = np.float32, ndmin = 2)


data_arr_act  = load_data(datafile_folder+datafile_name_act+datafile_ext)
assert data_arr_act.shape == (numDataPts_act, dimInput+dimOutput), data_arr_act.shape

data_input_act, data_output_act = np.split(data_arr_act, [dimInput], axis=1)
x_data_act = np.linspace(0, 10, 15000)
y_data_act = np.empty([numE_act, numN_act, dimOutput])
for i in range(numN_act):
    for j in range(numE_act):
        y_data_act[j, i, :] = data_output_act[numE_act*i+j, :]

data_arr_pred  = load_data(datafile_folder+datafile_name_pred+datafile_ext)
assert data_arr_pred.shape == (numDataPts_pred, dimInput+dimOutput), data_arr_pred.shape

data_input_pred, data_output_pred = np.split(data_arr_pred, [dimInput], axis=1)
x_data_pred = np.linspace(0, 10, 11)
y_data_pred = np.empty([numE_pred, numN_pred, dimOutput])
for i in range(numN_pred):
    for j in range(numE_pred):
        y_data_pred[j, i, :] = data_output_pred[numE_pred*i+j, :]

print(y_data_act.shape)
print(x_data_pred.shape)

############################################################################################################

############################################### Defining Animation function ###############################################

def plotter(x_data_act, y_data_act, x_data_pred, y_data_pred, fig_nm, subplot_rows, subplot_cols, y_max):
    ''' x_data- numpy array of shape (1, numN_act)
        y_data- numpy array of shape (numE_act, numN_act, dimOutput)
        fig_nm- string '''

    # create a figure with subplots
    fig, ax = plt.subplots(subplot_rows, subplot_cols)

    # plotting on each subplot
    cnt = 0
    l_act=[0]*numE_act
    l_pred=[0]*numE_pred
    clrs = ['k', 'r', 'b', 'g', 'c']
    subTitles = ["\u03B2"+r"$_{1}$", "\u03B2"+r"$_{2}$", "\u03B2"+r"$_{3}$", "\u03B4"+r"$_{21}$", "\u03B4"+r"$_{31}$", "\u03B4"+r"$_{32}$"]
    for i in range(subplot_rows):
        for j in range(subplot_cols):
            for k in range(numE_act):
                l_act[k], = ax[i][j].plot(x_data_act.reshape((numN_act)), y_data_act[k, :, cnt].reshape((numN_act)), ls="-", lw=0.3, color=clrs[k])
                l_pred[k], = ax[i][j].plot(x_data_pred.reshape((numN_pred)), y_data_pred[k, :, cnt].reshape((numN_pred)), marker="x", ms=3, lw=0, color=clrs[k])

            matplotlib.axes.Axes.text(ax[i][j], x=0.5, y=0.9, s=subTitles[cnt], horizontalalignment='center',verticalalignment='center', transform=ax[i][j].transAxes)

            # axes limit settings
            if cnt<3:
                y_max = 5*(cnt+2)
                major_ticks = np.arange(0, y_max, 5)
                minor_ticks = np.arange(0, y_max, 0.5)
            else:
                y_max=1
                major_ticks = np.arange(0, y_max, 0.2)
                minor_ticks = np.arange(0, y_max, 0.02)
            ax[i][j].set_ylim(0, y_max)
            ax[i][j].set_xlim(0, 10)

            # y-axis ticks setting    
            ax[i][j].set_yticks(major_ticks)
            ax[i][j].set_yticks(minor_ticks, minor=True)
            ax[i][j].grid(which='both', axis='y')
            ax[i][j].grid(which='minor', axis='y', alpha=0.2)
            ax[i][j].grid(which='major', axis='y', alpha=0.5)
            ax[i][j].tick_params(labelsize="x-small")

            cnt += 1

    fig.suptitle("Output Values vs \u03B7 for different values of \u03B4" + r"$_{0}$", fontsize=16)
    # fig.legend(tuple(l_act+l_pred) , ("0.1", "0.2", "0.3", "0.4", "0.5", "0.1", "0.2", "0.3", "0.4", "0.5"), loc = 'lower center', prop={'size': 7}, ncol=5, labelspacing=0. )
    fig.legend(tuple(l_act+[l_pred[0]]) , ("0.1", "0.2", "0.3", "0.4", "0.5", "Predicted Values"), loc = 'lower center', ncol=6, prop={'size': 7}, labelspacing=0. )
    plt.savefig(fig_nm + ".svg")
    # plt.savefig(fig_nm + ".png", dpi=1200)

    # closing all figures
    plt.close('all')

############################################################################################################


plotter(x_data_act, y_data_act, x_data_pred, y_data_pred, plt_folder+plt_name, num_subplot_rows, num_subplot_cols, 1)
