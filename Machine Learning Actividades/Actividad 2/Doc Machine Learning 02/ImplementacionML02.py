# COMPUTACIÓN BLANDA - Sistemas y Computación
# -------------------------------------------
# Introducción a numpy
# -------------------------------------------
# Lección 02

import numpy as np
# Para empezar, vamos a crear dos arrays
# Matriz a
a = np.arange(9).reshape(3,3)
print('A=\n', a, '\n')
# Matriz b, creada a partir de la matriz a
b = a*3
print('B =\n', b)
#----------------------------------------------
# APILAMIENTO HORIZONTAL
# Si axis=1, el apilamiento es horizontal
print('Matrices origen:')
print('a =\n', a, '\n')
print('b =\n', b, '\n')
# Apilamiento horizontal
print('Apilamiento horizontal =\n', np.hstack((a,b)),'\n')
# APILAMIENTO HORIZONTAL - Variante
# Utilización de la función: concatenate()
# Matrices origen
print('A =\n', a, '\n')
print('B =\n', b, '\n')
# Apilamiento horizontal
print( 'Apilamiento horizontal con concatenate = \n',
np.concatenate((a,b), axis=1))
#--------------------------------------------------------------
# APILAMIENTO VERTICAL
# Si axis=0, el apilamiento es vertical
print('Matrices origen:')
print('A =\n', a, '\n')
print('B =\n', b, '\n')
# Apilamiento vertical
print( 'Apilamiento vertical =\n', np.vstack((a,b)), '\n')
# APILAMIENTO VERTICAL - Variante
# Utilización de la función: concatenate()
# Matrices origen
print('Matrices origen:')
print('A =\n', a, '\n')
print('B =\n', b, '\n')
# Apilamiento vertical
print( 'Apilamiento vertical con concatenate =\n',
 np.concatenate((a,b), axis=0) )
#----------------------------------------------------
 # APILAMIENTO EN PROFUNDIDAD
# En el apilamiento en profundidad, se crean bloques utilizando
# parejas de datos tomados de las dos matrices
# Matrices origen
print('Matrices origen:')
print('A =\n', a, '\n')
print('B =\n', b, '\n')
# Apilamiento en profundidad
print( 'Apilamiento en profundidad =\n', np.dstack((a,b)))
#---------------------------------------------------------------
# APILAMIENTO POR COLUMNAS
# El apilamiento por columnas es similar a hstack()
# Se apilan las columnas, de izquierda a derecha, y tomándolas
# de los bloques definidos en la matriz
# Matrices origen
print('Matrices origen:')
print('A =\n', a, '\n')
print('B =\n', b, '\n')
# Apilamiento vertical
print( 'Apilamiento por columnas =\n',np.column_stack((a,b)) )
#-------------------------------------------------------------------
# APILAMIENTO POR FILAS
# El apilamiento por fila es similar a vstack()
# Se apilan las filas, de arriba hacia abajo, y tomándolas
# de los bloques definidos en la matriz
# Matrices origen
print('Matrices origen:')
print('a =\n', a, '\n')
print('b =\n', b, '\n')
# Apilamiento vertical
print( 'Apilamiento por filas =\n', np.row_stack((a,b)) )
#-------------------------------------------------------------------
#DIVISIÓN DE ARRAYS
# DIVISIÓN HORIZONTAL
print(a, '\n')
# El código resultante divide una matriz a lo largo de su eje horizontal
# en tres piezas del mismo tamaño y forma:}
print('Array con división horizontal =\n', np.hsplit(a, 3), '\n')
# El mismo efecto se consigue con split() y utilizando una bandera a 1
print('Array con división horizontal, uso de split() =\n', np.split(a, 3, axis=1),'\n')

 # DIVISIÓN VERTICAL
print(a, '\n')
# La función vsplit divide el array a lo largo del eje vertical:
print('División Vertical = \n', np.vsplit(a, 3), '\n')
# El mismo efecto se consigue con split() y utilizando una bandera a 0
print('Array con división vertical, uso de split() =\n', np.split(a, 3, axis=0))

# DIVISIÓN EN PROFUNDIDAD
# La función dsplit, como era de esperarse, realiza división
# en profundidad dentro del array
# Para ilustrar con un ejemplo, utilizaremos una matriz de rango tres
c = np.arange(27).reshape(3, 3, 3)
print(c, '\n')
# Se realiza la división
print('División en profundidad =\n', np.dsplit(c,3), '\n')

# El atributo ndim calcula el número de dimensiones
print(b, '\n')
print('ndim: ', b.ndim)

# El atributo size calcula el número de elementos
print(b, '\n')
print('size: ', b.size)

# El atributo itemsize obtiene el número de bytes por cada
# elemento en el array
print('itemsize: ', b.itemsize)

# El atributo nbytes calcula el número total de bytes del array
print(b, '\n')
print('nbytes: ', b.nbytes, '\n')
# Es equivalente a la siguiente operación
print('nbytes equivalente: ', b.size * b.itemsize)

# El atributo T tiene el mismo efecto que la transpuesta de la matriz
b.resize(6,4)
print(b, '\n')
print('Transpuesta: ', b.T)

# Los números complejos en numpy se representan con j
b = np.array([1.j + 1, 2.j + 3])
print('Complejo: \n', b)

# El atributo real nos da la parte real del array,
# o el array en sí mismo si solo contiene números reales
print('real: ', b.real, '\n')
# El atributo imag contiene la parte imaginaria del array
print('imaginario: ', b.imag)

# Si el array contiene números complejos, entonces el tipo de datos
# se convierte automáticamente a complejo
print(b.dtype)
