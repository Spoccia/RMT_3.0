"""
Thi  scrit  staqrt to   read and load the data  for extracting  RMT3.0
"""

from config import configMocap as cnf
import numpy as np
import math
from numba import jit, prange
from scipy.spatial import distance
import scipy.stats as stats
import matplotlib.pyplot as plt


def compute_distance_matrix(locations):
    N = locations.shape[0] # numero di variates
    distance_matrix = np.zeros((N,N))

    for i in prange(N):
        point_i = locations[i]
        for j in prange(N):
            distance_matrix[i][j] = distance.euclidean(point_i, locations[j])
            #parallelDist(point_i,distance_matrix,i,locations)
    distance_matrix = (distance_matrix - np.min(distance_matrix)) / np.ptp(distance_matrix)
    return distance_matrix


def prepare_gaussians(sigma, mustprint):
    mu = 0
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    #print(sigma.shape)
    smoothing_curve= np.zeros(sigma.shape[0])
    smoothing_curve = stats.norm.pdf(x, mu, sigma[0])
    for i in range(sigma.shape[0]):
        smoothing_curve[:,i] = stats.norm.pdf(x[:,i], mu, sigma[i])
        if(mustprint != 0):
            plt.plot(x, smoothing_curve[i])
    if (mustprint != 0):
        plt.legend(sigma)
        plt.show()
    return smoothing_curve
# load  dependency file

def normalizzarighe(H):
    for i in range(H.shape[0]):
        H[i] = H[i]/sum(H[i])
    return H

def computeDependencyScale( H, sigma ):
    N = H.shape[0] # numero di variates
    W = math.ceil(4 * sigma)
    # crea un vettore di smoothing 1 riga  e 2*W colonne
    gaussFunc = np.zeros(2*W)
    acc = 0
    for j in range(2 * W):
        gaussFunc[j] = math.exp(-(j - W - 1) ** 2 / (2 * sigma ** 2))
        acc = acc + gaussFunc[j]
    gaussFunc = gaussFunc/acc
    Ho = H
    H = normalizzarighe(H)
    HT = normalizzarighe(Ho.transpose())


"""
@jit
def parallelDist(pointA,distance_matrix, i, locations):
    N = locations.shape[0]
    for j in range(N):
        distance_matrix[i][j] = distance.euclidean(pointA, locations[j])
"""


#read location path
location_path = cnf.dataset_path + cnf.location_path + cnf.location_filename
locations = np.loadtxt(location_path, dtype=float, delimiter=',')
distance_matrix = compute_distance_matrix(locations)
#print(distance_matrix)
# Produce vector smoothing dependency scale NB : per se vuoi 5 scale  devi produrre 6 scale  su cui fare differenze di scale
stepT = np.arange(0,1+ 1/cnf.scaletime,1/cnf.scaletime)
stepD = np.arange(0,1+ 1/cnf.scaledpd,1/cnf.scaledpd)
ktime = 2**(stepT) # step of scale for smoothing time
kdepd = 2**(stepD) # step of scale for smoothing dependency

scale_sigmaT = cnf.sigmaT * ktime
scale_sigmaD = cnf.sigmaD * kdepd #cnf.sigmaD/(cnf.scaledpd+1))
smoothing_curve_dep = prepare_gaussians(scale_sigmaD, 0) #[0:2], 0)
smoothing_curve_time = prepare_gaussians(scale_sigmaT, 0)
#print(scale_sigmaD)
"""per ogni scala faccio uno smoothing della matrice delle dipendenze e poi la si taglia  creando 
la finestra di smoothing 
non esistono  univariate features nel primo octave
"""

dpd_conv_matrix = 1 * (distance_matrix <= cnf.dpd_threshold)
computeDependencyScale(dpd_conv_matrix, cnf.sigmaD)
print(dpd_conv_matrix)

# calcolo le maschere per   fare lo smoothing lungo le variates per ogni octave Ci sono dpdN  smoothing  window



# calcolo la matrice delle distanze tra le posizioni dei sensori(variate)

