# Lab-convolucion-y-correlacion
## Descripcion
En el presente laboratorio se implementa código para realizar una convolucion entre un sistema y una señal, también se analiza la correlación entre dos señales y se aplica la transformada de Fourier para pasar la señal del dominio del tiempo al dominio de la frecuencia.
> [!TIP]
>Librerias necesarias:
>```
> import wfdb
>import seaborn as sns
>import matplotlib.pyplot as plt
>import numpy as np
>rom scipy.signal import welch
> ```
## Calcular la convolución
*describir que es la convolucion
En el ejemplo se almacenó la informacion del sistema en un vector y la de la señal en otro, posteriormente se usa la función de la libreria numpy "convolve" para obtener el resultado de la convolución
```
vectorh = [5,6,0,0,8,0,7]
vectorx= [1,1,1,8,1,6,7,8,4,2]

vectory=np.convolve(vectorx, vectorh, mode= 'full')
print("LAURA y[n]")
print(vectory)

```
![image](https://github.com/user-attachments/assets/2d9125c9-d2bf-4bb2-a2c2-fc7dc163f3d7)

Para graficar la convolucion se rellena la matriz ejex con el numero de datos de la convolución, despues usando las funciones vlines y scatter se hacen las barras que tienen el valor y dado por la convolucion
```
ejex=[]
for i in range(len(vectory)):
    ejex.append(i)
#grafica
fig, graf = plt.subplots()
graf.vlines(x=ejex, ymin=0, ymax=vectory, colors='b', linestyle='dashed', linewidth=2)  # Líneas verticales
graf.scatter(ejex, vectory, color='r', zorder=3)  # Puntos en los extremos de las líneas

plt.xlabel("n")
plt.ylabel("y[n]")
plt.title("Gráfica de la convolución")

plt.grid()
plt.show()
```
![image](https://github.com/user-attachments/assets/d1391f57-912f-4b9a-a4aa-45147077a721)
![image](https://github.com/user-attachments/assets/cf69bbc2-e15b-4065-b1c9-a15b2c36feac)
![image](https://github.com/user-attachments/assets/7c747481-37e3-4988-b658-b585a4db7fc4)

## Correlación señal senoidal y cosenoidal
El código genera dos señales discretas, una coseno (x1) y otra seno (x2), ambas con una frecuencia de 100 Hz y muestreadas cada 1.25 ms. Luego, calcula la correlación cruzada entre ellas utilizando np.correlate, lo que permite analizar su similitud en diferentes desfases. Finalmente, se genera un vector de desfases (lags) que indica cómo se desplaza x2 respecto a x1, permitiendo identificar la alineación óptima entre ambas señales, que en este caso ocurre con un desfase de 90° debido a la diferencia de fase entre el seno y el coseno.
```
# Definición de las señales
Ts = 1.25e-3  # Período de muestreo
n = np.arange(0, 9)  # Valores de n

x1 = np.cos(2 * np.pi * 100 * n * Ts)  # Señal x1[nTs]
x2 = np.sin(2 * np.pi * 100 * n * Ts)  # Señal x2[nTs]

# Cálculo de la correlación cruzada
correlacion = np.correlate(x1, x2, mode="full")#Calcula la correlación cruzada de las señales
lags = np.arange(-len(x1) + 1, len(x1))  # Valores de desfase
```
![image](https://github.com/user-attachments/assets/2e9728e0-3b6b-4861-a7e0-035555b710b0)
![image](https://github.com/user-attachments/assets/d735adbe-4419-4ed0-8336-27a7b8096f06)

## Caracterización de la señal en función del tiempo y estadísticos descriptivos
Caracterización en función del tiempo
```
# Convertir muestras a tiempo en segundos
tiempo = np.arange(senal.shape[0]) / frecuencia
```
> [!TIP]
>Para un análisis mas sencillo de la señal tome solo un canal de la electromiografía
>```
># Seleccionar un solo canal (por ejemplo, el primero)
>canal_idx = 0  # Cambia este índice para elegir otro canal
>senal_canal = senal[:, canal_idx]
> ```
Se calcularon los estadísticos descriptivos usando numpy 
```
# Calcular estadísticas con NumPy
media = np.mean(senal_canal)  # Media
desviacion = np.std(senal_canal, ddof=1)  # Desviación estándar
coef_variacion = (desviacion / media) * 100  # Coeficiente de variación (%)}
```
Para el histograma se usó la librería seaborn
```
sns.histplot(senal_canal, bins=60, color='purple', edgecolor='black', alpha=0.7, kde=True)
```











