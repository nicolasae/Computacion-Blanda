# COMPUTACIÓN BLANDA - Sistemas y Computación
# -------------------------------------------
# Introducción a numpy
# -------------------------------------------
# Lección 01

import numpy as np

#Creo el vector de 8 elementos con index 0
a = np.arange(8)

#Imprimo el arreglo
print('A = ',a,'\n')

#Imprimir los tipos de datos del vector, int64
print('Tipo de A =', a.dtype, '\n')

#Muestra la dimension del vector, en este caso debe ser 1
print('Dimensión de a =', a.ndim, '\n')

# Se calcula el número de elementos del vector a
print('Número de elementos del array a =', a.shape,'\n')

#---------------------------------------------------
#Creo un areglo multidimensional, se crea con la funcion array
print('Arreglos multidimensionales:\n')
n = np.array([np.arange(3), np.arange(3)])
#Imprimo el array n
print(n,'\n')

 # Seleccionando elementos de un array, tienen que ser de la misma dimension
a = np.array([[1,2,6], [3,4,5]])
print('A =\n', a,'\n')
# Elementos individuales
print('A[0,0] =', a[0,0], '\n')
print('A[0,1] =', a[0,1], '\n')
print('A[1,0] =', a[1,0], '\n')
print('A[1,1] =', a[1,1], '\n')
print('A[0,2] =', a[0,2], '\n')

# Crea un array con 11 elementos, desde 0 hasta 10
b = np.arange(11)
print('B =', b, '\n')
# Muestra los elementos desde 0 hasta 9. Imprime desde 0 hasta 8
print('B[0:10] = ', b[0:10],'\n')
# Muestra desde 3 hasta 7. Imprime desde 3 hasta 6
print('B[2,5] =', b[2:5],'\n')
# Mostrando todos los elementos, desde el 0 hasta el 8, de uno en uno
print('B[0:9:1] =', b[0:11:1], '\n')
# El mismo ejemplo, pero omitiendo el número 0 al principio, el cual no es necesario aquí
print('B[:11:1] =', b[:11:1], '\n')
# Mostrando los números, de dos en dos
print('B[0:11:2] =', b[0:11:2], '\n')
# Mostrando los números, de tres en tres
print('B[0:11:3] =', b[0:11:3],'\n')

# Si utilizamos un incremento negativo, el array se muestra en orden inverso
# El problema es que no muestra el valor 0
print('B[11:0:-2] =', b[11:0:-2], '\n')
# Si se omiten los valores de índice, el resultado es preciso
print('B[::-3] =', b[::-3],'\n')

# Utilización de arreglos multidimensionales, el rango debe estar dentro de la mutliplicaciones en este caso de 2x2x3=12
c= np.arange(12).reshape(2,2,3)
# La instrucción reshape genera una matriz con 2 bloques, 2 filas y 3 columnas
# El número total de elementos es de 28 (generados por arange)
print('C =\n', c,'\n')
#-----------------------------------------------------------------------
# Acceso individual a los elementos del array
# Elemento en el bloque 0, fila 1, columna 2
print('C[0,1,2] =', c[0,1,2], '\n')
# Elemento en el bloque 0, fila 2, columna 1
print('C[1,1,1] =', c[1,1,1], '\n')
# Elemento en el bloque 0, fila 1, columna 1
print('C[0,1,1] =', c[0,1,1],'\n')

#-----------------------------------------------------------------------------
# Mostraremos como generalizar una selección
# Primero elegimos el componente en la fila 0, columna 0, del bloque 0
print('C[0,0,0] =',c[0,0,0], '\n')
# A continuación, elegimos el componente en la fila 0, columna, pero del bloque 1
print('C[1,0,0] =', c[1,0,0], '\n')
# Para elegir SIMULTANEAMENTE ambos elementos, lo hacemos utilizando dos puntos
print('C[:,0,0] =', c[:,0,0],'\n')

 # Si escribimos: b[0]
# Habremos elegido el primer bloque, pero habríamos omitido las filas y las columnas
# En tal caso, numpy toma todas las filas y columnas del bloque 0
print('C[0] =\n', c[0],'\n')

# Otra forma de representar b[0] es: b[0, :, :]
# Los dos puntos sin ningún valor, indican que se utilizarán todos los términos disponibles
# En este caso, todas las filas y todas las columnas
print('C[0,:,:] =\n', c[0,:,:],'\n')
# Cuando se utiliza la notación de : a derecha o a izquierda, se puede reemplazar por ...
# El ejemplo anterior se puede escribir así:
print('C[0, ...] =\n', c[0, ...],'\n')

# Si queremos la fila 1 en el bloque 0 (sin que importen las columnas), se tiene:
print('C[0,1] =', c[0,1])

#------------------------------------------------------------------------------------------
# El resultado de una selección puede utilizar luego para un cálculo posterior
# Se obtiene la fila 1 del bloque 0 (como en ejemplo anterior)
# y se asigna dicha respuesta a la variable z
z = c[0,1]
print('z =', z, '\n')
# En este caso, la variable z toma el valor: [4 5 6 7]
# Si ahora queremos tomar de dicha respuesta los valores de 2 en 2, se tiene:
print('Z[::2] =', z[::2],'\n')
# El ejercicio anterior se puede combinar en una expresión única, así:
print('C[C,1,::2] =', c[0,1,::2],'\n')


# Imprime todas las columnas, independientemente de los bloques y filas
print(c, '\n')
print('C[:,:,1] =\n', c[:,:,1], '\n')
# Variante de notación (simplificada)
print('C[...,1] =\n', c[...,1],'\n')


# Si queremos seleccionar todas las filas 2, independientemente
# de los bloques y columnas, se tiene:
print(c, '\n')
print('C[:,1] =', c[:,1],'\n')

# En el siguiente ejemplo seleccionmos la columna 0 del bloque 0
print(c, '\n')
print('C[0,:,1] =', c[0,:,0],'\n')
#------------------------------------------------------------------------------

# Si queremos seleccionar la última columna del primer bloque, tenemos:
print('C[0,:,-1] =', c[0,:,-1])
# La expresión ::-1 invierte todos los valores que se hubieran seleccionado
print('C[0, ::-1, -1] =', c[0, ::-1, -1])
# Si en lugar de invertir la columna, quisieramos imprimir sus
# valores de 2 en 2, tendríamos:
print('C[0, ::2, -1] =', c[0, ::2, -1])


# El array original
print(c, '\n-----------------------\n')
# Esta instrucción invierte los bloques
print(c[::-1],'\n')
#-------------------------------------------------------------------------------
# La instrucción: ravel(), de-construye el efecto de la instrucción: reshape
# Este es el array c en su estado matricial
print('Matriz C =\n', c, '\n--------------------------\n')
# Con ravel() se genera un vector a partir de la matriz
print('Vector C= ', c.ravel(),'\n')
# La instrucción: flatten() es similar a ravel()
# La diferencia es que flatten genera un nuevo espacio de memoria
print('Vector C con flatten =', c.flatten(),'\n')

#--------------------------------------------------------------------------------
# Se puede cambiar la estructura de una matriz con la instrucción: shape
# Transformamos la matriz en 6 filas x 4 columnas
c.shape = (3,4)
print('C(3x4) =',c,'\n')
#------------------------------------------------------------------------------
# A partir de la matriz que acaba de ser generada, vamos a mostrar
# como se construye la transpuesta de la matriz
# Matriz original
print('Matriz C=\n', c, '\n------------------------\n')
# Matri transpuesta
print('Transpuesta de C =\n', c.transpose(), '\n------------------------\n')
