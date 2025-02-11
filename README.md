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










