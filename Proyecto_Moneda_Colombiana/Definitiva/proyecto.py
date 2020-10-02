import os
from utils import DATA_DIR, CHART_DIR
import numpy as np
np.seterr(all='ignore')
import scipy as sp
import matplotlib.pyplot as plt
data = np.genfromtxt(os.path.join(DATA_DIR, "Datos_Historicos_COP.tsv"),
                     delimiter="\t")

data = np.array(data, dtype=np.float64)
print(data[:10])
print(data.shape)

colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-.', '--', ':', '-']
x = data[:, 0]
y = data[:, 1]

print("Número de entradas incorrectas:", np.sum(np.isnan(y)))
print(x)
print(y)
intervalo_tiempo = int(34 / 2)
conversion_temporal = 52 * 2
prediccion = 5000


x = x[~np.isnan(y)]
y = y[~np.isnan(y)]
def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    ''' dibujar datos de entrada '''
    plt.figure(num=None, figsize=(10, 8))
    plt.clf()
    plt.scatter(x, y, s=10)
    plt.title("Precio historico COP")

    plt.xlabel("Tiempo")

    plt.ylabel("Tasa de intercambio COP a USD")

    plt.xticks(
        [w  * conversion_temporal for w in range(intervalo_tiempo + 10)],
        ['A\n' +str(int(w*2+1990)) for w in range(intervalo_tiempo + 10)])

    if models:
        if mx is None:
            mx = np.linspace(0, x[-1], 1000)

        for model, style, color in zip(models, linestyles, colors):
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)
        plt.legend(["d=%i" % m.order for m in models], loc="upper left")
    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    plt.savefig(fname)
plot_models(x, y, None, os.path.join(CHART_DIR, "Grafico_inicial.png"))

fp1, res1, rank1, sv1, rcond1 = np.polyfit(x, y, 1, full=True)
print("Parámetros del modelo fp1: %s" % fp1)
print("Error del modelo fp1:", res1)
f1 = np.poly1d(fp1)
fp2, res2, rank2, sv2, rcond2 = np.polyfit(x, y, 2, full=True)

print("Parámetros del modelo fp2: %s" % fp2)
print("Error del modelo fp2:", res2)
f2 = np.poly1d(fp2)
f3 = np.poly1d(np.polyfit(x, y, 3))
f5 = np.poly1d(np.polyfit(x, y, 5))
f10 = np.poly1d(np.polyfit(x, y, 10))
f100 = np.poly1d(np.polyfit(x, y, 70))
plot_models(x, y, [f1], os.path.join(CHART_DIR, "Grafica_lineal.png"))
plot_models(x, y, [f1, f2], os.path.join(CHART_DIR, "Grafica_Grado2.png"))
plot_models(
    x, y, [f1, f2, f3, f10, f100], os.path.join(CHART_DIR,"Grafica_polinomial.png"))
inflexion = 3.5 * conversion_temporal
xa = x[:int(inflexion)]
ya = y[:int(inflexion)]
xb = x[int(inflexion):]
yb = y[int(inflexion):]

fa = np.poly1d(np.polyfit(xa, ya, 1))
fb = np.poly1d(np.polyfit(xb, yb, 1))

plot_models(x, y, [fa, fb], os.path.join(CHART_DIR, "1400_01_05.png"))

def error(f, x, y):
    return np.sum((f(x) - y) ** 2)
print("Errores para el conjunto completo de datos:")
for f in [f1, f2, f3,f5, f10]:
    print("Error d=%i: %f" % (f.order, error(f, x, y)))

print("Errores solamente después del punto de inflexión")
for f in [f1, f2, f3,f5, f10]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))

print("Error de inflexión=%f" % (error(fa, xa, ya) + error(fb, xb, yb)))

plot_models(
    x, y, [f3,f5, f10],
    os.path.join(CHART_DIR, "1400_01_06.png"),
    mx=np.linspace(0 *conversion_temporal, (intervalo_tiempo + 4) * conversion_temporal , 100),
    ymax=prediccion + 3000, xmin=0 * conversion_temporal)


print("Entrenamiento de datos únicamente despúes del punto de inflexión")
fb1 = fb
fb2 = np.poly1d(np.polyfit(xb, yb, 2))
fb3 = np.poly1d(np.polyfit(xb, yb, 3))
fb5 = np.poly1d(np.polyfit(xb, yb, 5))
fb10 = np.poly1d(np.polyfit(xb, yb, 10))
fb100 = np.poly1d(np.polyfit(xb, yb, 70))

print("Errores después del punto de inflexión")
for f in [fb1, fb2, fb3,fb5, fb10]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))

plot_models(
    x, y, [fb3,fb5, fb10],
    os.path.join(CHART_DIR, "1400_01_07.png"),
    mx=np.linspace(0 * conversion_temporal, (intervalo_tiempo+4)*conversion_temporal, 100),
    ymax=prediccion + 3000 , xmin=0 * conversion_temporal)

frac = 0.3
split_idx = int(frac * len(xb))
shuffled = np.random.permutation(list(range(len(xb))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])
fbt1 = np.poly1d(np.polyfit(xb[train], yb[train], 1))
fbt2 = np.poly1d(np.polyfit(xb[train], yb[train], 2))
print("fbt2(x)= \n%s" % fbt2)
print("fbt2(x)-100,000= \n%s" % (fbt2-prediccion))

fbt3 = np.poly1d(np.polyfit(xb[train], yb[train], 3))
fbt5 = np.poly1d(np.polyfit(xb[train], yb[train], 5))
fbt10 = np.poly1d(np.polyfit(xb[train], yb[train], 10))
fbt100 = np.poly1d(np.polyfit(xb[train], yb[train], 70))
print("Prueba de error para después del punto de inflexión")
for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
    print("Error d=%i: %f" % (f.order, error(f, xb[test], yb[test])))

plot_models(
    x, y, [fbt3, fbt5, fbt10],
    os.path.join(CHART_DIR, "1400_01_08.png"),
    mx=np.linspace(0 *conversion_temporal, (intervalo_tiempo+4)*conversion_temporal, 100),
    ymax=prediccion + 3000 , xmin=0 *conversion_temporal)
from scipy.optimize import fsolve
print(fbt2)
print(fbt2 - prediccion)
alcanzado_max = fsolve(fbt10 - prediccion, x0=1700) / (conversion_temporal)
ano = alcanzado_max[0] * 2 +1990
print("\nLa moneda alcanzará el valor de conversion de "+ str(prediccion) + " en el año %f" %
      ano)