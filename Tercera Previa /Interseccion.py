import numpy as np
import skfuzzy as sk
import matplotlib.pyplot as plt

x = np.arange(0,11, 1)

bajo = sk.trimf(x, [0,0,5])
medio = sk.trimf(x, [0,5,10])

#Configuracion
plt.figure()
plt.plot(x,bajo,'b',linewidth = 1.5, label='Bajo')
plt.plot(x,medio,'r',linewidth = 1.5, label='Medio')

#Ajustes gráfico
plt.title('Función Intersección(máximo)')
plt.ylabel('Membresía')
plt.xlabel("Velocidad (Kilometros Por Hora)")
plt.legend(loc='center right',bbox_to_anchor = (1.25,0.5),ncol=1,fancybox=True,shadow = True)

plt.axvline(x=0,ymin=0,ymax=10,color="g",linestyle='-.')
plt.axvline(x=1,ymin=0,ymax=10,color="g",linestyle='-.')
plt.axvline(x=2,ymin=0,ymax=10,color="g",linestyle='-.')
plt.axvline(x=3,ymin=0,ymax=10,color="g",linestyle='-.')
plt.axvline(x=4,ymin=0,ymax=10,color="g",linestyle='-.')
plt.axvline(x=5,ymin=0,ymax=10,color="g",linestyle='-.')
plt.axvline(x=6,ymin=0,ymax=10,color="g",linestyle='-.')
plt.axvline(x=7,ymin=0,ymax=10,color="g",linestyle='-.')
plt.axvline(x=8,ymin=0,ymax=10,color="g",linestyle='-.')
plt.axvline(x=9,ymin=0,ymax=10,color="g",linestyle='-.')
plt.axvline(x=10,ymin=0,ymax=10,color="g",linestyle='-.')

plt.plot(2.5,0.5,marker='o', markersize=10, color='g')
plt.plot(10,0,marker='o', markersize=10, color='g')

plt.savefig('Interseccion')

print(sk.fuzzy_and(x,bajo,x,medio))


