
"""
This  is a configuratio n file to define  where we caqn  load  the dtasets to read, \
    where is the output  files and  eventually all the configurations we need
"""

import os


dataset_path = 'E:'+os.sep+ 'dataset_RMT3_0'+ os.sep
location_path = 'location' + os.sep
location_filename = 'LocationMatrixMocap.csv'
datapath = 'data'+ os.sep
destination_path= ""



depd_file_name = ""
start_filename = ""

distance_measure = ""
dpd_threshold = 0.07 #DeGaussianThres TRESHOLD with the normalization of the distance matrix should be  between 0 and 1
# execution parameters
scaletime = 5     #:  Number of maximum scale over Time in an octave teh consecutive windows  ar smoothed  of a difference of gaussian  of (sigma to 2 sigma )/ st
scaledpd = 5     #:  Number of maximum scale over Dependency
n_octaveDPD = 0
n_octaveTIME = 2
sigmaT = 4
sigmaD = 0.5 #DeSigmaDepd = 0.5;

sigmaNT = 0.0 # sipresuppone che  il dto  abbia subito uno smoothing iniziale di una gaussiana di 0.5, lo metto a 0 per ora

n_bin_descriptors= 125

#### indice di partenza octave e scale
stmin=0;
sdmin=0;
otmin=0;
odmin=0;