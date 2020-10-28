import numpy as np
import skfuzzy as sk
import matplotlib.pyplot as plt


#Se define en x un array 
x = np.arange(0,11,0.6)
#Se define la varible para la gaussiana
vd_gaussiana_bell = sk.gbellmf (x,2,3,5)

#Se grafica
plt.figure()
plt.plot(x, vd_gaussiana_bell, 'b', linewidth = 1.5, label = 'Servicio')

plt.title ('Calidad del Servicio en un Restaurante')
plt.ylabel('Membresía')
plt.xlabel ('Nivel de Servicio')
plt.legend(loc='center right', bbox_to_anchor =(1.25, 0.5), ncol =1, fancybox = True, shadow = True )


plt.savefig('FuncionGaussiana-Bell')
