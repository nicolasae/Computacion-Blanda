from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import pandas as pd

# Datos de entrenamiento
# --------------------------------------------------------------
data = pd.read_csv("train.csv")

X = data.iloc[:,0:20] # Columnas independientes
y = data.iloc[:,-1] # Elija la última columna para la función de destino

print("data.iloc => X: \n\n", X, "\n")
print("data.iloc => y: \n\n", y, "\n")


# In[29]:


# Modelo de clasificación
# ------------------------------------------------------------
modelo = ExtraTreesClassifier()

# Entrenamiento
modelo.fit(X,y)

print(modelo.feature_importances_) # Utilizar clase incorporada


# In[30]:


# Importancia de las características de los
# clasificadores basados en árboles.

# Se traza un gráfico de importancia de las características para una
# mejor visualización
# --------------------------------------------------------------

modelo = ExtraTreesClassifier()
modelo.fit(X,y)

# Cálculo valores importancia
# --------------------------------------------------------------
valores_importancia = pd.Series(modelo.feature_importances_, index=X.
columns)
valores_importancia.nlargest(5).plot(kind='barh')
plt.show()
