import numpy as np
import matplotlib.pyplot as plt 
from numpy import mat
spike=np.loadtxt("ALIF_freq1.txt") 
data=mat(spike).nonzero()
plt.scatter(data[0], data[1], s=1, c='navy', alpha=0.05)
plt.show()
