# -- coding: utf-8 --
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch

vectorh = [5,6,0,0,8,3,2]    
vectorx= [1,0,3,4,3,9,6,0,6,6]  
            
vectory=np.convolve(vectorx, vectorh, mode= 'full')
print("CATALINA y[n]")
print(vectory)
ejex=[]

for i in range(len(vectory)):
    ejex.append(i)

#grafica

fig, graf = plt.subplots()
graf.vlines(x=ejex, ymin=0, ymax=vectory, colors='b', linestyle='dashed', linewidth=2)  # Líneas verticales
graf.scatter(ejex, vectory, color='r', zorder=3)  # Puntos en los extremos de las líneas

plt.xlabel("n")
plt.ylabel("y[n]")
plt.title("Gráfica de la convolución CATALINA")

plt.grid()
plt.show()

vectorh = [5,6,0,0,7,6,1]    
vectorx= [1,0,0,7,4,3,8,2,1,5]  
            
vectory=np.convolve(vectorx, vectorh, mode= 'full')
print("PABLO y[n]")
print(vectory)
ejex=[]

for i in range(len(vectory)):
    ejex.append(i)

#grafica

fig, graf = plt.subplots()
graf.vlines(x=ejex, ymin=0, ymax=vectory, colors='b', linestyle='dashed', linewidth=2)  # Líneas verticales
graf.scatter(ejex, vectory, color='r', zorder=3)  # Puntos en los extremos de las líneas

plt.xlabel("n")
plt.ylabel("y[n]")
plt.title("Gráfica de la convolución PABLO")

plt.grid()
plt.show()

vectorh = [5,6,0,0,8,0,7]    
vectorx= [1,1,1,8,1,6,7,8,4,2]  
            
vectory=np.convolve(vectorx, vectorh, mode= 'full')
print("LAURA y[n]")
print(vectory)
ejex=[]

for i in range(len(vectory)):
    ejex.append(i)

#grafica

fig, graf = plt.subplots()
graf.vlines(x=ejex, ymin=0, ymax=vectory, colors='b', linestyle='dashed', linewidth=2)  # Líneas verticales
graf.scatter(ejex, vectory, color='r', zorder=3)  # Puntos en los extremos de las líneas

plt.xlabel("n")
plt.ylabel("y[n]")
plt.title("Gráfica de la convolución LAURA")

plt.grid()
plt.show()

# Definición de las señales
Ts = 1.25e-3  # Período de muestreo
n = np.arange(0, 9)  # Valores de n

x1 = np.cos(2 * np.pi * 100 * n * Ts)  # Señal x1[nTs]
x2 = np.sin(2 * np.pi * 100 * n * Ts)  # Señal x2[nTs]

# Cálculo de la correlación cruzada
correlacion = np.correlate(x1, x2, mode="full")#Calcula la correlación cruzada de las señales
lags = np.arange(-len(x1) + 1, len(x1))  # Valores de desfase
print("Corelación cruzada")
print(correlacion)
# Gráficas
fig, axs = plt.subplots(3, 1, figsize=(10, 8))#figura con tres subgraficos cada uno en una fila,figsize=(10, 8) defina el tamaño de la figura

# Gráfica de x1[n]
axs[0].stem(n, x1)
axs[0].set_title("Señal x1[nTs] = cos(2π100nTs)")
axs[0].set_xlabel("n")
axs[0].set_ylabel("Amplitud")
axs[0].grid()

# Gráfica de x2[n]
axs[1].stem(n, x2)
axs[1].set_title("Señal x2[nTs] = sin(2π100nTs)")
axs[1].set_xlabel("n")
axs[1].set_ylabel("Amplitud")
axs[1].grid()

# Gráfica de la correlación
axs[2].stem(lags, correlacion)
axs[2].set_title("Correlación cruzada de x1[n] y x2[n]")
axs[2].set_xlabel("Desfase (lags)")
axs[2].set_ylabel("Amplitud")
axs[2].grid()

plt.tight_layout()
plt.show()

# Nombre del archivo del registro
emg1 = 'session1_participant1_gesture10_trial1'

# Leer los datos guardados en el archivo .dat y .hea
try:
    record = wfdb.rdrecord(emg1)
except FileNotFoundError:
    print(f"El archivo {emg1} no se encontró. Verifica la ruta o el nombre del archivo.")
    exit()

# Extraer la señal, etiquetas y frecuencia de muestreo
senal = record.p_signal
etiquetas = record.sig_name
frecuencia = record.fs

# Mostrar la frecuencia de muestreo
print(f"Frecuencia de muestreo: {frecuencia} Hz")
# Convertir muestras a tiempo en segundos
tiempo = np.arange(senal.shape[0]) / frecuencia

# Seleccionar un solo canal (por ejemplo, el primero)
canal_idx = 0  # Cambia este índice para elegir otro canal
senal_canal = senal[:, canal_idx]

# Graficar la señal
plt.figure(figsize=(10, 4))
plt.plot(tiempo, senal_canal, label=f'Canal: {etiquetas[canal_idx]}')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Señal EMG canal 1')
plt.legend()
plt.grid()
plt.show()

# ---- TRANSFORMADA DE FOURIER ----
N = len(senal_canal)  # Número de muestras
fft_result = np.fft.fft(senal_canal)  # Transformada de Fourier
frecuencias = np.fft.fftfreq(N, d=1/frecuencia)  # Eje de frecuencias
fft_result = np.abs(fft_result[:N//2])  # Magnitud de la FFT (solo parte positiva) dada la simetria
frecuencias = frecuencias[:N//2]  # Filtrar solo las frecuencias positivas

# ---- DENSIDAD ESPECTRAL DE POTENCIA (PSD) ----
frecs_psd, psd = welch(senal_canal, fs=frecuencia, nperseg=1024)

# ---- GRAFICAR FFT Y PSD ----
plt.figure(figsize=(12, 6))

# Gráfico de la Transformada de Fourier
plt.subplot(2, 1, 1)
plt.plot(frecuencias, fft_result, color='r', linewidth=1.5)
plt.title(f"Transformada de Fourier - {etiquetas[canal_idx]}")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True, linestyle="--", alpha=0.7)

# Gráfico de la Densidad Espectral de Potencia (PSD)
plt.subplot(2, 1, 2)
plt.semilogy(frecs_psd, psd, color='b', linewidth=1.5)
plt.title(f"Densidad Espectral de Potencia (PSD) - {etiquetas[canal_idx]}")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad de Potencia")
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("ecg_fft_psd.png", dpi=300, bbox_inches='tight')
plt.show()

#Histograma
plt.figure(figsize=(10, 6))
sns.histplot(senal_canal, bins=60, color='purple', edgecolor='black', alpha=0.7, kde=True)
# Configurar el gráfico
plt.title('Histograma y Densidad de Probabilidad de la Señal EMG')
plt.xlabel('Amplitud (mV)')
plt.ylabel('Densidad de Probabilidad')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show() 

# Seleccionar un canal (por ejemplo, el primer canal)
canal_idx = 0  # Cambia este índice si quieres otro canal
senal_canal = senal[:, canal_idx]  # Extraer solo el canal seleccionado
etiqueta = etiquetas[canal_idx]  # Nombre del canal

# Calcular estadísticas con NumPy
media = np.mean(senal_canal)  # Media
desviacion = np.std(senal_canal, ddof=1)  # Desviación estándar
coef_variacion = (desviacion / media) * 100  # Coeficiente de variación (%)

# Mostrar resultados
print("ESTADISTICOS EN FUNCION DEL TIEMPO:")
print(f"Canal {etiqueta}:")
print(f"  Media: {media:.3f}")
print(f"  Desviación estándar: {desviacion:.3f}")
print(f"  Coeficiente de variación: {coef_variacion:.2f}%")

# Mostrar información básica
print("ESTADISTICOS EN FUNCION DE LA FRECUENCIA")
print(f"\nCanal seleccionado: {etiqueta}")
print(f"Frecuencia de muestreo: {frecuencia} Hz")
print(f"Número total de muestras: {senal_canal.shape[0]}")
#calcular estadisticos para frecuencia
frecuencia_media = np.sum(frecuencias * fft_result) / np.sum(fft_result)
desviacion_frecuencia = np.sqrt(np.sum(fft_result * (frecuencias - frecuencia_media)**2) / np.sum(fft_result))
acumulado = np.cumsum(fft_result)  
total = np.sum(fft_result)
frecuencia_mediana = frecuencias[np.where(acumulado >=total / 2)[0][0]]
plt.figure(figsize=(10, 6))
plt.hist(frecuencias, bins=50, weights=fft_result, color='blue', edgecolor='black', alpha=0.7)
print(f"Frecuencia Media: {frecuencia_media:.2f} Hz")
print(f"Frecuencia Mediana: {frecuencia_mediana:.2f} Hz")
print(f"Desviación Estándar de Frecuencia: {desviacion_frecuencia:.2f} Hz")

#grafica del histograma
plt.title('Histograma de Frecuencias de la Señal')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad Espectral')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
