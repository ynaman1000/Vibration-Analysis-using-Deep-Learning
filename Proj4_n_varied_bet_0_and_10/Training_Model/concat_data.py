# -*- coding: utf-8 -*-
import torch
import numpy as np
from random import shuffle

############################################### Manual Setting ###############################################

datafile1_folder = "Test_Data/"
datafile1_name = "test_data1"    # name of data file
datafile1_ext = ".csv"
datafile2_folder = "Test_Data/"
datafile2_name = "test_data2"    # name of data file
datafile2_ext = ".csv"

############################################################################################################


############################################### Appending Data of file2 to file1 ###############################################

# function for appending data
def append_data(fl_nm1, fl_nm2):
    ''' Appends data of datafile2 at the end of datafile1 '''
    data_file1 = open(fl_nm1, 'a')
    data_file2 = open(fl_nm2, 'r')
    for row in data_file2.readlines():
        data_file1.write(row)
    data_file1.close()
    data_file2.close()   

append_data(datafile1_folder+datafile1_name+datafile1_ext, datafile2_folder+datafile2_name+datafile2_ext)
############################################################################################################
