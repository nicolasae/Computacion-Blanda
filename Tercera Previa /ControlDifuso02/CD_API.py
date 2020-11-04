import warnings
warnings.filterwarnings('ignore')

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as crtl

#Se crean los objetos antecedentes y consecuente a partir de las 
#variables del universo y las funciones de membresía, varibale 
#linguistica que antecede 
calidad = crtl.Antecedent(np.arange(0,11,1),'calidad')
servicio = crtl.Antecedent(np.arange(0,11,1),'servicio')
propina = crtl.Consequent(np.arange(0,26,1),'propina')

#La poblacioń de la función de membresía automática es posible con 
# .automf ( 3,5 o 7 )
calidad.automf(3)
servicio.automf(3)

#Las funciones de membresía personalizadas se pueden construir interactivamente 
#con la API Pyhtonic
propina['bajo'] = fuzz.trimf(propina.universe, [0,0,13])
propina['medio'] = fuzz.trimf(propina.universe, [0,13,25])
propina['alto'] = fuzz.trimf(propina.universe, [13,25,25])

#Visualización con .view()
calidad['average'].view()
servicio.view()
propina.view()

#Creación de las reglas
regla1 = crtl.Rule(calidad['poor'] | servicio['poor'],propina['bajo'])
regla2 = crtl.Rule(servicio['average'],propina['medio'])
regla3 = crtl.Rule(servicio['good'] | calidad['good'],propina['alto'])

#Visualización de la regla1
regla1.view()

#Generación del simulador
control_propina = crtl.ControlSystem([regla1,regla2,regla3])
asignacion_propina = crtl.ControlSystemSimulation(control_propina)

#Pasar entradas al ControlSystem usando etiquetas 'Antecedent' con 
#Pythonica API. nota: si quiere pasar muchas entradas a la vez, usar .inputs
# (dict_of:data)
asignacion_propina.input['calidad'] = 6.5
asignacion_propina.input['servicio'] = 9.8

#Se obtiene el valor 
asignacion_propina.compute()

#Se muestra la información
print("Valor de la propina: ")
print(asignacion_propina.output['propina'])

#Se muestra la curva de asignación de propina
propina.view( sim = asignacion_propina)



