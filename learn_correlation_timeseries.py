from config import configMocap as cnf
import numpy as np
import math
from numba import jit, prange
from scipy.spatial import distance
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path = cnf.dataset_path + cnf.datapath + '1.csv'

data = np.loadtxt(path, dtype=float, delimiter=',')
lable = ["root-1","root-2","root-3","root-4","root-5","root-6",
         "lowerback-1","lowerback-2","lowerback-3","upperback-1","upperback-2","upperback-3","thorax-1",
         "thorax-2","thorax-3","lowerneck-1","lowerneck-2","lowerneck-3","upperneck-1","upperneck-2","upperneck-3",
         "head-1","head-2","head-3","rclavicle-1","rclavicle-2","rhumerus-1","rhumerus-2","rhumerus-3",
         "rradius","rwrist","rhand-1","rhand-2","rfingers","rthumb-1","rthumb-2","lclavicle-1",
         "lclavicle-2","lhumerus-1","lhumerus-2","lhumerus-3","lradius","lwrist","lhand-1","lhand-2","lfingers",
         "lthumb-1","lthumb-2","rfemur-1","rfemur-2","rfemur-3","rtibia","rfoot-1","rfoot-2","rtoes","lfemur-1",
         "lfemur-2","lfemur-3","ltibia","lfoot-1","lfoot-2","ltoes"]
dataframe = pd.DataFrame(data, columns=lable)
corr = dataframe.corr(method='pearson')
sns.heatmap(1 * (corr>0.6))
plt.show()
#sns.clustermap(corr)
print(data)


