#Nicolás Aguirre Espinosa

import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------------#
#[IN 50]

data = np.genfromtxt("web_traffic.tsv", delimiter="\t")
#print(data[:10], '\n')

#print(data.shape)

#-------------------------------------------------------------------------------------------------------------------------------#
#[IN 51]
x = data[:,0]
y = data[:,1]
#print(x, '\n')
#print(y, '\n')

#[IN 52]
#print(x.ndim, '\n')
#print(y.ndim, '\n')
#print(x.shape, '\n')
#print(y.shape, '\n')

#[IN 53]
#print(np.sum(np.isnan(y)),'\n')

#[IN 54]
#print(x.shape,)
#print(y.shape, '\n')

x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

#print(x.shape,)
#print(x.shape, '\n')

#[IN 55]
plt.scatter(x, y, s=6)
plt.title("Tráfico Web ")
plt.xlabel("Tiempo")
plt.ylabel("Accesos/Hra")
plt.xticks([w*7*24 for w in range(20)],['semana %i' % w for w in range(20)])
plt.autoscale(tight=True)
plt.grid(True, linestyle='-', color='0.9')
plt.show()
